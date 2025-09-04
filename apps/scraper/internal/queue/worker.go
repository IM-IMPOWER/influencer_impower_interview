// AIDEV-NOTE: Worker implementation for processing scraping jobs
// Handles job execution with timeout, retry logic, and error handling
package queue

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/go-redis/redis/v8"
	"kol-scraper/pkg/logger"
)

// Worker processes jobs from the queue
type Worker struct {
	ID       string
	queue    *JobQueue
	logger   *logger.Logger
	stopChan chan struct{}
	isActive bool
}

// Start starts the worker loop
func (w *Worker) Start(ctx context.Context) {
	w.logger.QueueLog("", "worker", "started", logger.Fields{
		"worker_id": w.ID,
	})

	for {
		select {
		case <-ctx.Done():
			w.logger.QueueLog("", "worker", "stopped_context", logger.Fields{
				"worker_id": w.ID,
			})
			return
		case <-w.stopChan:
			w.logger.QueueLog("", "worker", "stopped_signal", logger.Fields{
				"worker_id": w.ID,
			})
			return
		default:
			w.processNextJob(ctx)
		}
	}
}

// processNextJob fetches and processes the next available job
func (w *Worker) processNextJob(ctx context.Context) {
	// AIDEV-NOTE: Try to get a job from the pending queue with timeout
	jobID, err := w.dequeueJob(ctx)
	if err != nil {
		if err != redis.Nil {
			w.logger.ErrorLog(err, "dequeue_job", logger.Fields{
				"worker_id": w.ID,
			})
		}
		// AIDEV-NOTE: Wait before trying again to avoid busy waiting
		time.Sleep(1 * time.Second)
		return
	}

	if jobID == "" {
		// AIDEV-NOTE: No jobs available, wait before checking again
		time.Sleep(1 * time.Second)
		return
	}

	// AIDEV-NOTE: Get job details
	job, err := w.queue.GetJob(ctx, jobID)
	if err != nil {
		w.logger.ErrorLog(err, "get_job", logger.Fields{
			"worker_id": w.ID,
			"job_id":    jobID,
		})
		return
	}

	// AIDEV-NOTE: Process the job with timeout and error handling
	w.executeJob(ctx, job)
}

// dequeueJob removes a job from the pending queue and adds it to running
func (w *Worker) dequeueJob(ctx context.Context) (string, error) {
	// AIDEV-NOTE: Use ZPOPMAX to get highest priority job atomically
	result, err := w.queue.redis.ZPopMax(ctx, QueueKeyPending, 1).Result()
	if err != nil {
		return "", err
	}

	if len(result) == 0 {
		return "", nil
	}

	jobID := result[0].Member.(string)

	// AIDEV-NOTE: Move job to running queue
	score := float64(time.Now().Unix())
	if err := w.queue.redis.ZAdd(ctx, QueueKeyRunning, &redis.Z{
		Score:  score,
		Member: jobID,
	}).Err(); err != nil {
		w.logger.ErrorLog(err, "move_to_running", logger.Fields{
			"worker_id": w.ID,
			"job_id":    jobID,
		})
		return "", err
	}

	return jobID, nil
}

// executeJob executes a job with timeout and retry logic
func (w *Worker) executeJob(ctx context.Context, job *Job) {
	w.isActive = true
	defer func() {
		w.isActive = false
	}()

	w.logger.QueueLog(job.ID, job.Type, "execution_started", logger.Fields{
		"worker_id": w.ID,
		"platform":  job.Platform,
		"username":  job.Username,
		"attempt":   job.RetryCount + 1,
	})

	// AIDEV-NOTE: Update job status to running
	if err := w.queue.UpdateJobStatus(ctx, job.ID, StatusRunning); err != nil {
		w.logger.ErrorLog(err, "update_job_status", logger.Fields{
			"job_id": job.ID,
			"status": StatusRunning,
		})
	}

	// AIDEV-NOTE: Create context with job timeout
	jobCtx, cancel := context.WithTimeout(ctx, job.Timeout)
	defer cancel()

	startTime := time.Now()
	var jobErr error

	// AIDEV-NOTE: Execute job with panic recovery
	func() {
		defer func() {
			if r := recover(); r != nil {
				jobErr = fmt.Errorf("job panicked: %v", r)
				w.logger.ErrorLog(jobErr, "job_panic", logger.Fields{
					"job_id":    job.ID,
					"worker_id": w.ID,
					"panic":     r,
				})
			}
		}()

		// AIDEV-NOTE: Get job handler and execute
		handler, exists := w.queue.jobHandlers[job.Type]
		if !exists {
			jobErr = fmt.Errorf("no handler for job type: %s", job.Type)
			return
		}

		jobErr = handler.Handle(jobCtx, job)
	}()

	duration := time.Since(startTime)

	// AIDEV-NOTE: Handle job completion or failure
	if jobErr != nil {
		w.handleJobFailure(ctx, job, jobErr, duration)
	} else {
		w.handleJobSuccess(ctx, job, duration)
	}
}

// handleJobSuccess handles successful job completion
func (w *Worker) handleJobSuccess(ctx context.Context, job *Job, duration time.Duration) {
	w.logger.QueueLog(job.ID, job.Type, "execution_completed", logger.Fields{
		"worker_id":       w.ID,
		"duration_ms":     duration.Milliseconds(),
		"platform":        job.Platform,
		"username":        job.Username,
	})

	// AIDEV-NOTE: Move job from running to completed
	pipe := w.queue.redis.Pipeline()
	pipe.ZRem(ctx, QueueKeyRunning, job.ID)
	pipe.ZAdd(ctx, QueueKeyCompleted, &redis.Z{
		Score:  float64(time.Now().Unix()),
		Member: job.ID,
	})

	if _, err := pipe.Exec(ctx); err != nil {
		w.logger.ErrorLog(err, "move_to_completed", logger.Fields{
			"job_id": job.ID,
		})
	}

	// AIDEV-NOTE: Update job status
	if err := w.queue.UpdateJobStatus(ctx, job.ID, StatusCompleted); err != nil {
		w.logger.ErrorLog(err, "update_job_status", logger.Fields{
			"job_id": job.ID,
			"status": StatusCompleted,
		})
	}

	// AIDEV-NOTE: Store job result
	result := &JobResult{
		JobID:     job.ID,
		Success:   true,
		Result:    job.Result,
		Duration:  duration,
		Timestamp: time.Now().UTC(),
	}

	w.storeJobResult(ctx, result)
}

// handleJobFailure handles job failure with retry logic
func (w *Worker) handleJobFailure(ctx context.Context, job *Job, jobErr error, duration time.Duration) {
	job.RetryCount++
	job.Error = jobErr.Error()

	w.logger.QueueLog(job.ID, job.Type, "execution_failed", logger.Fields{
		"worker_id":       w.ID,
		"duration_ms":     duration.Milliseconds(),
		"platform":        job.Platform,
		"username":        job.Username,
		"error":           jobErr.Error(),
		"retry_count":     job.RetryCount,
		"max_retries":     job.MaxRetries,
	})

	// AIDEV-NOTE: Check if we should retry the job
	if job.RetryCount < job.MaxRetries {
		w.scheduleRetry(ctx, job)
	} else {
		w.markJobAsFailed(ctx, job, duration)
	}
}

// scheduleRetry schedules a job for retry with exponential backoff
func (w *Worker) scheduleRetry(ctx context.Context, job *Job) {
	// AIDEV-NOTE: Calculate exponential backoff delay
	backoffSeconds := int64(1 << (job.RetryCount - 1)) // 1, 2, 4, 8, 16...
	if backoffSeconds > 300 { // Max 5 minutes
		backoffSeconds = 300
	}

	nextRetry := time.Now().Add(time.Duration(backoffSeconds) * time.Second)
	job.NextRetry = &nextRetry
	job.Status = StatusRetry

	// AIDEV-NOTE: Move job from running to retry queue
	pipe := w.queue.redis.Pipeline()
	pipe.ZRem(ctx, QueueKeyRunning, job.ID)
	pipe.ZAdd(ctx, QueueKeyRetry, &redis.Z{
		Score:  float64(nextRetry.Unix()),
		Member: job.ID,
	})

	if _, err := pipe.Exec(ctx); err != nil {
		w.logger.ErrorLog(err, "schedule_retry", logger.Fields{
			"job_id":     job.ID,
			"next_retry": nextRetry,
		})
		return
	}

	// AIDEV-NOTE: Update job data
	if err := w.queue.updateJob(ctx, job); err != nil {
		w.logger.ErrorLog(err, "update_retry_job", logger.Fields{
			"job_id": job.ID,
		})
	}

	w.logger.QueueLog(job.ID, job.Type, "scheduled_retry", logger.Fields{
		"next_retry":   nextRetry,
		"retry_count":  job.RetryCount,
		"backoff_secs": backoffSeconds,
	})
}

// markJobAsFailed marks a job as permanently failed
func (w *Worker) markJobAsFailed(ctx context.Context, job *Job, duration time.Duration) {
	// AIDEV-NOTE: Move job from running to failed queue
	pipe := w.queue.redis.Pipeline()
	pipe.ZRem(ctx, QueueKeyRunning, job.ID)
	pipe.ZAdd(ctx, QueueKeyFailed, &redis.Z{
		Score:  float64(time.Now().Unix()),
		Member: job.ID,
	})

	if _, err := pipe.Exec(ctx); err != nil {
		w.logger.ErrorLog(err, "move_to_failed", logger.Fields{
			"job_id": job.ID,
		})
	}

	// AIDEV-NOTE: Update job status
	if err := w.queue.UpdateJobStatus(ctx, job.ID, StatusFailed); err != nil {
		w.logger.ErrorLog(err, "update_job_status", logger.Fields{
			"job_id": job.ID,
			"status": StatusFailed,
		})
	}

	// AIDEV-NOTE: Store job result
	result := &JobResult{
		JobID:     job.ID,
		Success:   false,
		Error:     job.Error,
		Duration:  duration,
		Timestamp: time.Now().UTC(),
	}

	w.storeJobResult(ctx, result)

	w.logger.QueueLog(job.ID, job.Type, "marked_failed", logger.Fields{
		"final_error":  job.Error,
		"retry_count":  job.RetryCount,
		"max_retries":  job.MaxRetries,
	})
}

// storeJobResult stores the job result in Redis
func (w *Worker) storeJobResult(ctx context.Context, result *JobResult) {
	resultData, err := json.Marshal(result)
	if err != nil {
		w.logger.ErrorLog(err, "serialize_job_result", logger.Fields{
			"job_id": result.JobID,
		})
		return
	}

	resultKey := ResultKeyPrefix + result.JobID
	if err := w.queue.redis.Set(ctx, resultKey, resultData, 24*time.Hour).Err(); err != nil {
		w.logger.ErrorLog(err, "store_job_result", logger.Fields{
			"job_id": result.JobID,
		})
	}
}

// IsActive returns whether the worker is currently processing a job
func (w *Worker) IsActive() bool {
	return w.isActive
}

// GetStats returns worker statistics
func (w *Worker) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"id":        w.ID,
		"is_active": w.isActive,
	}
}