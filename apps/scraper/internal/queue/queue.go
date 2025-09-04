// AIDEV-NOTE: Job queue system for managing scraping tasks
// Provides Redis-backed job queue with worker management and retry logic
package queue

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/google/uuid"
	"kol-scraper/internal/models"
	"kol-scraper/pkg/config"
	"kol-scraper/pkg/logger"
)

// JobQueue manages scraping jobs with Redis backend
type JobQueue struct {
	redis       *redis.Client
	config      *config.Config
	logger      *logger.Logger
	workers     []*Worker
	workerWg    sync.WaitGroup
	stopChan    chan struct{}
	jobHandlers map[string]JobHandler
}

// Job represents a scraping job
type Job struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Platform    string                 `json:"platform"`
	Username    string                 `json:"username"`
	Priority    int                    `json:"priority"`
	Params      map[string]interface{} `json:"params"`
	CreatedAt   time.Time              `json:"created_at"`
	Timeout     time.Duration          `json:"timeout"`
	MaxRetries  int                    `json:"max_retries"`
	RetryCount  int                    `json:"retry_count"`
	Status      string                 `json:"status"`
	Error       string                 `json:"error,omitempty"`
	Result      interface{}            `json:"result,omitempty"`
	StartedAt   *time.Time             `json:"started_at,omitempty"`
	CompletedAt *time.Time             `json:"completed_at,omitempty"`
}

// JobHandler defines the interface for job processing
type JobHandler interface {
	Handle(ctx context.Context, job *Job) error
}

// JobResult represents the result of job processing
type JobResult struct {
	JobID     string      `json:"job_id"`
	Success   bool        `json:"success"`
	Result    interface{} `json:"result,omitempty"`
	Error     string      `json:"error,omitempty"`
	Duration  time.Duration `json:"duration"`
	Timestamp time.Time   `json:"timestamp"`
}

// Redis keys for different job states
const (
	QueueKeyPending   = "queue:pending"
	QueueKeyRunning   = "queue:running"
	QueueKeyCompleted = "queue:completed"
	QueueKeyFailed    = "queue:failed"
	QueueKeyRetry     = "queue:retry"
	JobKeyPrefix      = "job:"
	ResultKeyPrefix   = "result:"
)

// Job types
const (
	JobTypeSingleScrape = "single_scrape"
	JobTypeBulkScrape   = "bulk_scrape"
	JobTypeUpdateMetrics = "update_metrics"
	JobTypeUpdateProfile = "update_profile"
)

// Job status constants
const (
	StatusPending   = "pending"
	StatusRunning   = "running"
	StatusCompleted = "completed"
	StatusFailed    = "failed"
	StatusRetry     = "retry"
	StatusCancelled = "cancelled"
)

// NewJobQueue creates a new job queue instance
func NewJobQueue(cfg *config.Config, log *logger.Logger) (*JobQueue, error) {
	// AIDEV-NOTE: Create Redis client with configuration
	opts, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		return nil, fmt.Errorf("invalid Redis URL: %w", err)
	}

	if cfg.RedisPassword != "" {
		opts.Password = cfg.RedisPassword
	}
	opts.DB = cfg.RedisDB

	client := redis.NewClient(opts)

	// AIDEV-NOTE: Test Redis connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	jq := &JobQueue{
		redis:       client,
		config:      cfg,
		logger:      log,
		stopChan:    make(chan struct{}),
		jobHandlers: make(map[string]JobHandler),
	}

	log.Info("Job queue initialized", "redis_db", cfg.RedisDB)
	return jq, nil
}

// RegisterHandler registers a job handler for a specific job type
func (jq *JobQueue) RegisterHandler(jobType string, handler JobHandler) {
	jq.jobHandlers[jobType] = handler
	jq.logger.QueueLog("", jobType, "handler_registered", logger.Fields{})
}

// StartWorkers starts the specified number of workers
func (jq *JobQueue) StartWorkers(ctx context.Context) error {
	if len(jq.workers) > 0 {
		return fmt.Errorf("workers already started")
	}

	// AIDEV-NOTE: Create and start workers
	for i := 0; i < jq.config.QueueWorkers; i++ {
		worker := &Worker{
			ID:       fmt.Sprintf("worker-%d", i+1),
			queue:    jq,
			logger:   jq.logger,
			stopChan: make(chan struct{}),
		}

		jq.workers = append(jq.workers, worker)
		
		jq.workerWg.Add(1)
		go func(w *Worker) {
			defer jq.workerWg.Done()
			w.Start(ctx)
		}(worker)
	}

	jq.logger.QueueLog("", "system", "workers_started", logger.Fields{
		"worker_count": len(jq.workers),
	})

	// AIDEV-NOTE: Start cleanup routine for expired jobs
	jq.workerWg.Add(1)
	go func() {
		defer jq.workerWg.Done()
		jq.cleanupRoutine(ctx)
	}()

	return nil
}

// StopWorkers stops all workers gracefully
func (jq *JobQueue) StopWorkers() error {
	if len(jq.workers) == 0 {
		return nil
	}

	jq.logger.QueueLog("", "system", "stopping_workers", logger.Fields{
		"worker_count": len(jq.workers),
	})

	// AIDEV-NOTE: Signal all workers to stop
	close(jq.stopChan)
	for _, worker := range jq.workers {
		close(worker.stopChan)
	}

	// AIDEV-NOTE: Wait for all workers to finish with timeout
	done := make(chan struct{})
	go func() {
		jq.workerWg.Wait()
		close(done)
	}()

	select {
	case <-done:
		jq.logger.QueueLog("", "system", "workers_stopped", logger.Fields{})
	case <-time.After(30 * time.Second):
		jq.logger.QueueLog("", "system", "workers_stop_timeout", logger.Fields{})
		return fmt.Errorf("workers did not stop within timeout")
	}

	jq.workers = nil
	return nil
}

// Close closes the job queue and Redis connection
func (jq *JobQueue) Close() error {
	if err := jq.StopWorkers(); err != nil {
		jq.logger.ErrorLog(err, "stop_workers", logger.Fields{})
	}

	if err := jq.redis.Close(); err != nil {
		return fmt.Errorf("failed to close Redis connection: %w", err)
	}

	jq.logger.QueueLog("", "system", "queue_closed", logger.Fields{})
	return nil
}

// EnqueueJob adds a new job to the queue
func (jq *JobQueue) EnqueueJob(ctx context.Context, job *Job) error {
	if job.ID == "" {
		job.ID = uuid.New().String()
	}

	job.CreatedAt = time.Now().UTC()
	job.Status = StatusPending

	// AIDEV-NOTE: Set default values
	if job.Timeout == 0 {
		job.Timeout = jq.config.JobTimeout
	}
	if job.MaxRetries == 0 {
		job.MaxRetries = jq.config.RetryAttempts
	}

	// AIDEV-NOTE: Validate job
	if err := jq.validateJob(job); err != nil {
		return fmt.Errorf("job validation failed: %w", err)
	}

	// AIDEV-NOTE: Serialize job data
	jobData, err := json.Marshal(job)
	if err != nil {
		return fmt.Errorf("failed to serialize job: %w", err)
	}

	pipe := jq.redis.Pipeline()

	// AIDEV-NOTE: Store job data with expiration
	jobKey := JobKeyPrefix + job.ID
	pipe.Set(ctx, jobKey, jobData, 24*time.Hour)

	// AIDEV-NOTE: Add job to pending queue with priority
	pipe.ZAdd(ctx, QueueKeyPending, &redis.Z{
		Score:  float64(job.Priority),
		Member: job.ID,
	})

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to enqueue job: %w", err)
	}

	jq.logger.QueueLog(job.ID, job.Type, "enqueued", logger.Fields{
		"platform": job.Platform,
		"username": job.Username,
		"priority": job.Priority,
	})

	return nil
}

// GetJob retrieves a job by ID
func (jq *JobQueue) GetJob(ctx context.Context, jobID string) (*Job, error) {
	jobKey := JobKeyPrefix + jobID
	jobData, err := jq.redis.Get(ctx, jobKey).Result()
	if err != nil {
		if err == redis.Nil {
			return nil, fmt.Errorf("job not found: %s", jobID)
		}
		return nil, fmt.Errorf("failed to get job: %w", err)
	}

	var job Job
	if err := json.Unmarshal([]byte(jobData), &job); err != nil {
		return nil, fmt.Errorf("failed to deserialize job: %w", err)
	}

	return &job, nil
}

// UpdateJobStatus updates the status of a job
func (jq *JobQueue) UpdateJobStatus(ctx context.Context, jobID, status string) error {
	job, err := jq.GetJob(ctx, jobID)
	if err != nil {
		return err
	}

	job.Status = status
	now := time.Now().UTC()

	switch status {
	case StatusRunning:
		job.StartedAt = &now
	case StatusCompleted, StatusFailed, StatusCancelled:
		job.CompletedAt = &now
	}

	return jq.updateJob(ctx, job)
}

// CancelJob cancels a pending job
func (jq *JobQueue) CancelJob(ctx context.Context, jobID string) error {
	job, err := jq.GetJob(ctx, jobID)
	if err != nil {
		return err
	}

	if job.Status != StatusPending {
		return fmt.Errorf("cannot cancel job in status: %s", job.Status)
	}

	// AIDEV-NOTE: Remove from pending queue and mark as cancelled
	pipe := jq.redis.Pipeline()
	pipe.ZRem(ctx, QueueKeyPending, jobID)
	pipe.ZAdd(ctx, QueueKeyFailed, &redis.Z{Score: float64(time.Now().Unix()), Member: jobID})

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to cancel job: %w", err)
	}

	job.Status = StatusCancelled
	now := time.Now().UTC()
	job.CompletedAt = &now

	if err := jq.updateJob(ctx, job); err != nil {
		return fmt.Errorf("failed to update cancelled job: %w", err)
	}

	jq.logger.QueueLog(jobID, job.Type, "cancelled", logger.Fields{})
	return nil
}

// GetQueueStats returns statistics about the job queue
func (jq *JobQueue) GetQueueStats(ctx context.Context) (map[string]interface{}, error) {
	pipe := jq.redis.Pipeline()

	pendingCmd := pipe.ZCard(ctx, QueueKeyPending)
	runningCmd := pipe.ZCard(ctx, QueueKeyRunning)
	completedCmd := pipe.ZCard(ctx, QueueKeyCompleted)
	failedCmd := pipe.ZCard(ctx, QueueKeyFailed)
	retryCmd := pipe.ZCard(ctx, QueueKeyRetry)

	if _, err := pipe.Exec(ctx); err != nil {
		return nil, fmt.Errorf("failed to get queue stats: %w", err)
	}

	stats := map[string]interface{}{
		"pending_jobs":     pendingCmd.Val(),
		"running_jobs":     runningCmd.Val(),
		"completed_jobs":   completedCmd.Val(),
		"failed_jobs":      failedCmd.Val(),
		"retry_jobs":       retryCmd.Val(),
		"total_workers":    len(jq.workers),
		"active_workers":   jq.getActiveWorkerCount(),
		"uptime_seconds":   time.Now().Unix(), // Simplified uptime
	}

	return stats, nil
}

// validateJob validates job data before enqueueing
func (jq *JobQueue) validateJob(job *Job) error {
	if job.Type == "" {
		return fmt.Errorf("job type is required")
	}

	if _, exists := jq.jobHandlers[job.Type]; !exists {
		return fmt.Errorf("no handler registered for job type: %s", job.Type)
	}

	if job.Platform == "" {
		return fmt.Errorf("platform is required")
	}

	if job.Username == "" && job.Type != JobTypeBulkScrape {
		return fmt.Errorf("username is required for non-bulk jobs")
	}

	if job.Priority < 0 || job.Priority > 100 {
		return fmt.Errorf("priority must be between 0 and 100")
	}

	return nil
}

// updateJob updates job data in Redis
func (jq *JobQueue) updateJob(ctx context.Context, job *Job) error {
	jobData, err := json.Marshal(job)
	if err != nil {
		return fmt.Errorf("failed to serialize job: %w", err)
	}

	jobKey := JobKeyPrefix + job.ID
	if err := jq.redis.Set(ctx, jobKey, jobData, 24*time.Hour).Err(); err != nil {
		return fmt.Errorf("failed to update job: %w", err)
	}

	return nil
}

// getActiveWorkerCount returns the number of active workers
func (jq *JobQueue) getActiveWorkerCount() int {
	// AIDEV-NOTE: This is a simplified implementation
	// In production, you might want to track worker states more precisely
	return len(jq.workers)
}

// cleanupRoutine runs periodic cleanup of expired jobs
func (jq *JobQueue) cleanupRoutine(ctx context.Context) {
	ticker := time.NewTicker(jq.config.CleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-jq.stopChan:
			return
		case <-ticker.C:
			jq.cleanup(ctx)
		}
	}
}

// cleanup removes expired completed and failed jobs
func (jq *JobQueue) cleanup(ctx context.Context) {
	cutoff := time.Now().Add(-24 * time.Hour).Unix()

	// AIDEV-NOTE: Remove old completed jobs
	removedCompleted, err := jq.redis.ZRemRangeByScore(ctx, QueueKeyCompleted, "0", fmt.Sprintf("%.0f", float64(cutoff))).Result()
	if err != nil {
		jq.logger.ErrorLog(err, "cleanup_completed_jobs", logger.Fields{})
	} else if removedCompleted > 0 {
		jq.logger.QueueLog("", "cleanup", "completed_jobs_removed", logger.Fields{
			"count": removedCompleted,
		})
	}

	// AIDEV-NOTE: Remove old failed jobs
	removedFailed, err := jq.redis.ZRemRangeByScore(ctx, QueueKeyFailed, "0", fmt.Sprintf("%.0f", float64(cutoff))).Result()
	if err != nil {
		jq.logger.ErrorLog(err, "cleanup_failed_jobs", logger.Fields{})
	} else if removedFailed > 0 {
		jq.logger.QueueLog("", "cleanup", "failed_jobs_removed", logger.Fields{
			"count": removedFailed,
		})
	}
}