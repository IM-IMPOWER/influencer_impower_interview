// AIDEV-NOTE: HTTP handlers for KOL scraper API endpoints
// Provides REST API for scraping operations, job management, and data queries
package handlers

import (
	"context"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"kol-scraper/internal/database"
	"kol-scraper/internal/fastapi"
	"kol-scraper/internal/models"
	"kol-scraper/internal/queue"
	"kol-scraper/internal/scrapers"
	"kol-scraper/pkg/logger"
)

// Handler contains dependencies for HTTP handlers
type Handler struct {
	db            *database.DB
	queue         *queue.JobQueue
	scraperMgr    *scrapers.Manager
	logger        *logger.Logger
	fastapiClient *fastapi.Client
}

// New creates a new handler instance with dependencies
func New(db *database.DB, jobQueue *queue.JobQueue, log *logger.Logger) *Handler {
	return &Handler{
		db:     db,
		queue:  jobQueue,
		logger: log,
	}
}

// SetScraperManager sets the scraper manager (called after initialization)
func (h *Handler) SetScraperManager(mgr *scrapers.Manager) {
	h.scraperMgr = mgr
}

// Response structures for API endpoints
type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
	Message string      `json:"message,omitempty"`
}

type ScrapeRequest struct {
	Priority  int                    `json:"priority,omitempty"`
	Timeout   int                    `json:"timeout,omitempty"`   // seconds
	Params    map[string]interface{} `json:"params,omitempty"`
}

type BulkScrapeRequest struct {
	Platform  string                   `json:"platform"`
	Usernames []string                 `json:"usernames"`
	Priority  int                      `json:"priority,omitempty"`
	Timeout   int                      `json:"timeout,omitempty"`
	Params    map[string]interface{}   `json:"params,omitempty"`
}

type UpdateMetricsRequest struct {
	Platform  string   `json:"platform"`
	Usernames []string `json:"usernames"`
	ForceRefresh bool  `json:"force_refresh,omitempty"`
}

// AIDEV-NOTE: Health and status endpoints

// HealthCheck returns service health status
func (h *Handler) HealthCheck(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	health := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().UTC(),
		"version":   "1.0.0",
		"service":   "kol-scraper",
	}

	// AIDEV-NOTE: Check database health
	if err := h.db.Health(ctx); err != nil {
		health["status"] = "unhealthy"
		health["database"] = "unhealthy"
		health["database_error"] = err.Error()
		c.JSON(http.StatusServiceUnavailable, APIResponse{
			Success: false,
			Data:    health,
			Error:   "Database health check failed",
		})
		return
	}
	health["database"] = "healthy"

	// AIDEV-NOTE: Check queue health
	if queueStats, err := h.queue.GetQueueStats(ctx); err != nil {
		health["status"] = "unhealthy"
		health["queue"] = "unhealthy"
		health["queue_error"] = err.Error()
	} else {
		health["queue"] = "healthy"
		health["queue_stats"] = queueStats
	}

	// AIDEV-NOTE: Check scraper health
	if h.scraperMgr != nil {
		health["scrapers"] = h.scraperMgr.HealthCheck(ctx)
		health["available_platforms"] = h.scraperMgr.GetAvailablePlatforms()
	}

	statusCode := http.StatusOK
	if health["status"] == "unhealthy" {
		statusCode = http.StatusServiceUnavailable
	}

	c.JSON(statusCode, APIResponse{
		Success: health["status"] == "healthy",
		Data:    health,
	})
}

// Metrics returns comprehensive service metrics including database optimizations
func (h *Handler) Metrics(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	metrics := map[string]interface{}{
		"timestamp": time.Now().UTC(),
		"service":   "kol-scraper",
		"version":   "1.0.0",
	}

	// AIDEV-NOTE: 250903170700 Enhanced database metrics with optimization components
	dbStats := h.db.GetStats()
	databaseMetrics := map[string]interface{}{
		"connection_pool": map[string]interface{}{
			"open_connections":     dbStats.OpenConnections,
			"in_use":              dbStats.InUse,
			"idle":                dbStats.Idle,
			"wait_count":          dbStats.WaitCount,
			"wait_duration_ms":    dbStats.WaitDuration.Milliseconds(),
			"max_idle_closed":     dbStats.MaxIdleClosed,
			"max_lifetime_closed": dbStats.MaxLifetimeClosed,
		},
	}

	// AIDEV-NOTE: 250903170701 Add pool manager metrics if available
	if poolMetrics := h.db.GetPoolMetrics(); poolMetrics != nil {
		databaseMetrics["pool_optimization"] = map[string]interface{}{
			"connections_opened":    poolMetrics.ConnectionsOpened,
			"connections_closed":    poolMetrics.ConnectionsClosed,
			"connections_in_use":    poolMetrics.ConnectionsInUse,
			"connections_idle":      poolMetrics.ConnectionsIdle,
			"slow_queries":         poolMetrics.SlowQueries,
			"failed_connections":   poolMetrics.FailedConnections,
			"last_health_check":    poolMetrics.LastHealthCheck,
			"health_check_duration_ms": poolMetrics.HealthCheckDuration.Milliseconds(),
			"circuit_breaker_trips": poolMetrics.CircuitBreakerTrips,
		}
	}

	// AIDEV-NOTE: 250903170702 Add query executor metrics if available
	if queryMetrics := h.db.GetQueryMetrics(); queryMetrics != nil {
		databaseMetrics["query_optimization"] = map[string]interface{}{
			"total_queries":       queryMetrics.TotalQueries,
			"slow_queries":        queryMetrics.SlowQueries,
			"failed_queries":      queryMetrics.FailedQueries,
			"cache_hits":          queryMetrics.CacheHits,
			"cache_misses":        queryMetrics.CacheMisses,
			"cache_hit_ratio":     h.calculateCacheHitRatio(queryMetrics.CacheHits, queryMetrics.CacheMisses),
			"prepared_statements": queryMetrics.PreparedStatements,
			"last_reset":          queryMetrics.LastReset,
		}

		// Add query latency details
		if len(queryMetrics.QueryLatencies) > 0 {
			latencies := make(map[string]interface{})
			for operation, duration := range queryMetrics.QueryLatencies {
				latencies[operation+"_ms"] = duration.Milliseconds()
			}
			databaseMetrics["query_latencies"] = latencies
		}
	}

	// AIDEV-NOTE: 250903170703 Add retry metrics if available
	if retryMetrics := h.db.GetRetryMetrics(); retryMetrics != nil {
		databaseMetrics["resilience"] = map[string]interface{}{
			"total_operations":     retryMetrics.TotalOperations,
			"retried_operations":   retryMetrics.RetriedOperations,
			"failed_operations":    retryMetrics.FailedOperations,
			"retry_success":        retryMetrics.RetrySuccess,
			"avg_retry_delay_ms":   retryMetrics.AvgRetryDelay.Milliseconds(),
			"circuit_breaker_trips": retryMetrics.CircuitBreakerTrips,
			"success_rate":         h.calculateSuccessRate(retryMetrics.TotalOperations, retryMetrics.FailedOperations),
			"retry_rate":           h.calculateRetryRate(retryMetrics.RetriedOperations, retryMetrics.TotalOperations),
		}

		// Add retry attempts by operation type
		if len(retryMetrics.RetryAttempts) > 0 {
			databaseMetrics["retry_attempts"] = retryMetrics.RetryAttempts
		}

		// Add error distribution
		if len(retryMetrics.ErrorDistribution) > 0 {
			databaseMetrics["error_distribution"] = retryMetrics.ErrorDistribution
		}
	}

	// AIDEV-NOTE: 250903170704 Add monitoring metrics if available
	if monitorMetrics := h.db.GetMonitoringMetrics(); monitorMetrics != nil {
		databaseMetrics["monitoring"] = map[string]interface{}{
			"database_size_bytes":   monitorMetrics.DatabaseSize,
			"deadlocks":            monitorMetrics.Deadlocks,
			"cache_hit_ratio":      monitorMetrics.CacheHitRatio,
			"replication_lag_ms":   monitorMetrics.ReplicationLag.Milliseconds(),
			"alerts_triggered":     monitorMetrics.AlertsTriggered,
			"last_collected":       monitorMetrics.LastCollected,
		}

		// Add table sizes
		if len(monitorMetrics.TableSizes) > 0 {
			databaseMetrics["table_sizes"] = monitorMetrics.TableSizes
		}

		// Add transaction stats
		if monitorMetrics.Transactions != nil {
			databaseMetrics["transactions"] = map[string]interface{}{
				"committed":     monitorMetrics.Transactions.Committed,
				"rolled_back":   monitorMetrics.Transactions.RolledBack,
				"deadlocks":     monitorMetrics.Transactions.DeadLocks,
				"conflict_rate": monitorMetrics.Transactions.ConflictRate,
			}
		}

		// Add connection stats
		if monitorMetrics.ConnectionStats != nil {
			databaseMetrics["connection_activity"] = map[string]interface{}{
				"active":             monitorMetrics.ConnectionStats.Active,
				"idle":               monitorMetrics.ConnectionStats.Idle,
				"idle_in_transaction": monitorMetrics.ConnectionStats.IdleInTrans,
				"waiting":            monitorMetrics.ConnectionStats.Waiting,
				"total":              monitorMetrics.ConnectionStats.Total,
			}
		}
	}

	metrics["database"] = databaseMetrics

	// AIDEV-NOTE: 250903170705 Add active alerts information
	if activeAlerts := h.db.GetActiveAlerts(); activeAlerts != nil {
		alerts := make([]map[string]interface{}, 0, len(activeAlerts))
		for _, alert := range activeAlerts {
			alerts = append(alerts, map[string]interface{}{
				"name":       alert.Rule.Name,
				"severity":   string(alert.Rule.Severity),
				"message":    alert.Message,
				"start_time": alert.StartTime,
				"resolved":   alert.Resolved,
			})
		}
		metrics["database_alerts"] = alerts
	}

	// AIDEV-NOTE: Get queue stats
	if queueStats, err := h.queue.GetQueueStats(ctx); err == nil {
		metrics["queue"] = queueStats
	}

	// AIDEV-NOTE: Get scraper stats
	if h.scraperMgr != nil {
		metrics["scrapers"] = h.scraperMgr.GetScraperInfo()
	}

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data:    metrics,
	})
}

// AIDEV-NOTE: 250903170706 Helper methods for metric calculations
func (h *Handler) calculateCacheHitRatio(hits, misses int64) float64 {
	total := hits + misses
	if total == 0 {
		return 0.0
	}
	return float64(hits) / float64(total)
}

func (h *Handler) calculateSuccessRate(total, failed int64) float64 {
	if total == 0 {
		return 1.0
	}
	successful := total - failed
	return float64(successful) / float64(total)
}

func (h *Handler) calculateRetryRate(retried, total int64) float64 {
	if total == 0 {
		return 0.0
	}
	return float64(retried) / float64(total)
}

// AIDEV-NOTE: Scraping endpoints

// ScrapeKOL starts scraping for a single KOL
func (h *Handler) ScrapeKOL(c *gin.Context) {
	platform := c.Param("platform")
	username := c.Param("username")

	if platform == "" || username == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Platform and username are required",
		})
		return
	}

	// AIDEV-NOTE: Validate platform support
	platformEnum := models.Platform(platform)
	if h.scraperMgr != nil && !h.scraperMgr.IsSupported(platformEnum) {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Platform not supported: " + platform,
		})
		return
	}

	// AIDEV-NOTE: Validate username format
	if h.scraperMgr != nil {
		if err := h.scraperMgr.ValidateUsername(platformEnum, username); err != nil {
			c.JSON(http.StatusBadRequest, APIResponse{
				Success: false,
				Error:   "Invalid username: " + err.Error(),
			})
			return
		}
	}

	var req ScrapeRequest
	if c.Request.ContentLength > 0 {
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, APIResponse{
				Success: false,
				Error:   "Invalid request body: " + err.Error(),
			})
			return
		}
	}

	// AIDEV-NOTE: Create scraping job
	job := &queue.Job{
		Type:       queue.JobTypeSingleScrape,
		Platform:   platform,
		Username:   username,
		Priority:   req.Priority,
		Params:     req.Params,
		MaxRetries: 3,
	}

	if req.Timeout > 0 {
		job.Timeout = time.Duration(req.Timeout) * time.Second
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	if err := h.queue.EnqueueJob(ctx, job); err != nil {
		h.logger.ErrorLog(err, "enqueue_scrape_job", logger.Fields{
			"platform": platform,
			"username": username,
		})
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   "Failed to enqueue job: " + err.Error(),
		})
		return
	}

	h.logger.HTTPLog(c.Request.Method, c.Request.URL.Path, http.StatusAccepted, logger.Fields{
		"job_id":   job.ID,
		"platform": platform,
		"username": username,
	})

	c.JSON(http.StatusAccepted, APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"job_id":   job.ID,
			"status":   "queued",
			"platform": platform,
			"username": username,
		},
		Message: "Scraping job queued successfully",
	})
}

// BulkScrape starts bulk scraping for multiple KOLs
func (h *Handler) BulkScrape(c *gin.Context) {
	var req BulkScrapeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Invalid request body: " + err.Error(),
		})
		return
	}

	if req.Platform == "" || len(req.Usernames) == 0 {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Platform and usernames are required",
		})
		return
	}

	if len(req.Usernames) > 100 {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Maximum 100 usernames allowed per bulk request",
		})
		return
	}

	// AIDEV-NOTE: Validate platform support
	platformEnum := models.Platform(req.Platform)
	if h.scraperMgr != nil && !h.scraperMgr.IsSupported(platformEnum) {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Platform not supported: " + req.Platform,
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	jobIDs := make([]string, 0, len(req.Usernames))
	failed := make([]string, 0)

	// AIDEV-NOTE: Create individual jobs for each username
	for _, username := range req.Usernames {
		username = strings.TrimSpace(username)
		if username == "" {
			continue
		}

		// AIDEV-NOTE: Validate username format
		if h.scraperMgr != nil {
			if err := h.scraperMgr.ValidateUsername(platformEnum, username); err != nil {
				failed = append(failed, username+": "+err.Error())
				continue
			}
		}

		job := &queue.Job{
			Type:       queue.JobTypeSingleScrape,
			Platform:   req.Platform,
			Username:   username,
			Priority:   req.Priority,
			Params:     req.Params,
			MaxRetries: 3,
		}

		if req.Timeout > 0 {
			job.Timeout = time.Duration(req.Timeout) * time.Second
		}

		if err := h.queue.EnqueueJob(ctx, job); err != nil {
			h.logger.ErrorLog(err, "enqueue_bulk_job", logger.Fields{
				"platform": req.Platform,
				"username": username,
			})
			failed = append(failed, username+": "+err.Error())
			continue
		}

		jobIDs = append(jobIDs, job.ID)
	}

	h.logger.HTTPLog(c.Request.Method, c.Request.URL.Path, http.StatusAccepted, logger.Fields{
		"platform":      req.Platform,
		"total_users":   len(req.Usernames),
		"queued_jobs":   len(jobIDs),
		"failed_jobs":   len(failed),
	})

	response := map[string]interface{}{
		"queued_jobs":   len(jobIDs),
		"failed_jobs":   len(failed),
		"job_ids":       jobIDs,
		"platform":      req.Platform,
	}

	if len(failed) > 0 {
		response["failures"] = failed
	}

	c.JSON(http.StatusAccepted, APIResponse{
		Success: len(jobIDs) > 0,
		Data:    response,
		Message: "Bulk scraping jobs queued",
	})
}

// ScrapeStatus returns the status of a scraping job
func (h *Handler) ScrapeStatus(c *gin.Context) {
	jobID := c.Param("job_id")
	if jobID == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Job ID is required",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	job, err := h.queue.GetJob(ctx, jobID)
	if err != nil {
		statusCode := http.StatusInternalServerError
		if strings.Contains(err.Error(), "not found") {
			statusCode = http.StatusNotFound
		}

		c.JSON(statusCode, APIResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	// AIDEV-NOTE: Calculate progress and estimate completion time
	progress := 0
	var estimatedCompletion *time.Time

	switch job.Status {
	case queue.StatusPending:
		progress = 0
	case queue.StatusRunning:
		progress = 50
		if job.StartedAt != nil {
			// AIDEV-NOTE: Rough estimate based on average job duration
			avgDuration := 2 * time.Minute
			estimated := job.StartedAt.Add(avgDuration)
			estimatedCompletion = &estimated
		}
	case queue.StatusCompleted, queue.StatusFailed, queue.StatusCancelled:
		progress = 100
	case queue.StatusRetry:
		progress = 25
	}

	response := map[string]interface{}{
		"job_id":         job.ID,
		"status":         job.Status,
		"progress":       progress,
		"platform":       job.Platform,
		"username":       job.Username,
		"created_at":     job.CreatedAt,
		"retry_count":    job.RetryCount,
		"max_retries":    job.MaxRetries,
	}

	if job.StartedAt != nil {
		response["started_at"] = *job.StartedAt
	}

	if job.CompletedAt != nil {
		response["completed_at"] = *job.CompletedAt
		if job.StartedAt != nil {
			duration := job.CompletedAt.Sub(*job.StartedAt)
			response["duration_seconds"] = duration.Seconds()
		}
	}

	if estimatedCompletion != nil {
		response["estimated_completion"] = *estimatedCompletion
	}

	if job.NextRetry != nil {
		response["next_retry"] = *job.NextRetry
	}

	if job.Error != "" {
		response["error"] = job.Error
	}

	if job.Result != nil {
		response["result"] = job.Result
	}

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data:    response,
	})
}

// CancelScrape cancels a pending scraping job
func (h *Handler) CancelScrape(c *gin.Context) {
	jobID := c.Param("job_id")
	if jobID == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Job ID is required",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	if err := h.queue.CancelJob(ctx, jobID); err != nil {
		statusCode := http.StatusInternalServerError
		if strings.Contains(err.Error(), "not found") {
			statusCode = http.StatusNotFound
		} else if strings.Contains(err.Error(), "cannot cancel") {
			statusCode = http.StatusConflict
		}

		c.JSON(statusCode, APIResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	h.logger.HTTPLog(c.Request.Method, c.Request.URL.Path, http.StatusOK, logger.Fields{
		"job_id": jobID,
	})

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"job_id": jobID,
			"status": "cancelled",
		},
		Message: "Job cancelled successfully",
	})
}

// AIDEV-NOTE: Data update endpoints

// UpdateMetrics triggers metrics update for specified KOLs
func (h *Handler) UpdateMetrics(c *gin.Context) {
	var req UpdateMetricsRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Invalid request body: " + err.Error(),
		})
		return
	}

	if req.Platform == "" || len(req.Usernames) == 0 {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Platform and usernames are required",
		})
		return
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Second)
	defer cancel()

	jobIDs := make([]string, 0, len(req.Usernames))

	// AIDEV-NOTE: Create metrics update jobs
	for _, username := range req.Usernames {
		username = strings.TrimSpace(username)
		if username == "" {
			continue
		}

		job := &queue.Job{
			Type:     queue.JobTypeUpdateMetrics,
			Platform: req.Platform,
			Username: username,
			Priority: 75, // High priority for updates
			Params: map[string]interface{}{
				"force_refresh": req.ForceRefresh,
			},
			MaxRetries: 2,
		}

		if err := h.queue.EnqueueJob(ctx, job); err != nil {
			h.logger.ErrorLog(err, "enqueue_update_job", logger.Fields{
				"platform": req.Platform,
				"username": username,
			})
			continue
		}

		jobIDs = append(jobIDs, job.ID)
	}

	c.JSON(http.StatusAccepted, APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"queued_jobs": len(jobIDs),
			"job_ids":     jobIDs,
			"platform":    req.Platform,
		},
		Message: "Metrics update jobs queued",
	})
}

// UpdateProfile triggers profile update for a specific KOL
func (h *Handler) UpdateProfile(c *gin.Context) {
	kolID := c.Param("kol_id")
	if kolID == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "KOL ID is required",
		})
		return
	}

	// AIDEV-NOTE: Get KOL info from database to create update job
	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	var kol models.KOL
	query := `SELECT username, platform FROM kols WHERE id = $1 AND is_active = true`
	if err := h.db.GetContext(ctx, &kol, query, kolID); err != nil {
		statusCode := http.StatusNotFound
		if !strings.Contains(err.Error(), "no rows") {
			statusCode = http.StatusInternalServerError
		}

		c.JSON(statusCode, APIResponse{
			Success: false,
			Error:   "KOL not found or database error",
		})
		return
	}

	job := &queue.Job{
		Type:     queue.JobTypeUpdateProfile,
		Platform: string(kol.Platform),
		Username: kol.Username,
		Priority: 80, // Higher priority for profile updates
		Params: map[string]interface{}{
			"kol_id": kolID,
		},
		MaxRetries: 2,
	}

	if err := h.queue.EnqueueJob(ctx, job); err != nil {
		h.logger.ErrorLog(err, "enqueue_profile_update", logger.Fields{
			"kol_id":   kolID,
			"platform": kol.Platform,
			"username": kol.Username,
		})
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   "Failed to enqueue profile update job",
		})
		return
	}

	c.JSON(http.StatusAccepted, APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"job_id":   job.ID,
			"kol_id":   kolID,
			"platform": kol.Platform,
			"username": kol.Username,
		},
		Message: "Profile update job queued",
	})
}

// UpdateContent triggers content update for a specific KOL
func (h *Handler) UpdateContent(c *gin.Context) {
	kolID := c.Param("kol_id")
	if kolID == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "KOL ID is required",
		})
		return
	}

	// AIDEV-NOTE: Similar to UpdateProfile but for content updates
	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	var kol models.KOL
	query := `SELECT username, platform FROM kols WHERE id = $1 AND is_active = true`
	if err := h.db.GetContext(ctx, &kol, query, kolID); err != nil {
		statusCode := http.StatusNotFound
		if !strings.Contains(err.Error(), "no rows") {
			statusCode = http.StatusInternalServerError
		}

		c.JSON(statusCode, APIResponse{
			Success: false,
			Error:   "KOL not found or database error",
		})
		return
	}

	job := &queue.Job{
		Type:     "update_content",
		Platform: string(kol.Platform),
		Username: kol.Username,
		Priority: 60, // Medium priority for content updates
		Params: map[string]interface{}{
			"kol_id": kolID,
		},
		MaxRetries: 2,
	}

	if err := h.queue.EnqueueJob(ctx, job); err != nil {
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   "Failed to enqueue content update job",
		})
		return
	}

	c.JSON(http.StatusAccepted, APIResponse{
		Success: true,
		Data: map[string]interface{}{
			"job_id":   job.ID,
			"kol_id":   kolID,
			"platform": kol.Platform,
			"username": kol.Username,
		},
		Message: "Content update job queued",
	})
}