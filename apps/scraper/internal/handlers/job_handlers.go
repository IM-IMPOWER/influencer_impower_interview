// AIDEV-NOTE: Job handlers for processing different types of scraping jobs
// Implements JobHandler interface for queue processing
package handlers

import (
	"context"
	"fmt"
	"time"

	"kol-scraper/internal/database"
	"kol-scraper/internal/models"
	"kol-scraper/internal/queue"
	"kol-scraper/internal/scrapers"
	"kol-scraper/pkg/logger"
)

// ScrapeJobHandler handles single KOL scraping jobs
type ScrapeJobHandler struct {
	db         *database.DB
	scraperMgr *scrapers.Manager
	logger     *logger.Logger
}

// NewScrapeJobHandler creates a new scrape job handler
func NewScrapeJobHandler(db *database.DB, scraperMgr *scrapers.Manager, logger *logger.Logger) *ScrapeJobHandler {
	return &ScrapeJobHandler{
		db:         db,
		scraperMgr: scraperMgr,
		logger:     logger,
	}
}

// Handle processes a single scraping job
func (h *ScrapeJobHandler) Handle(ctx context.Context, job *queue.Job) error {
	platform := models.Platform(job.Platform)
	username := job.Username

	h.logger.QueueLog(job.ID, job.Type, "processing_started", logger.Fields{
		"platform": platform,
		"username": username,
	})

	// AIDEV-NOTE: Validate platform support
	if !h.scraperMgr.IsSupported(platform) {
		return fmt.Errorf("platform %s not supported", platform)
	}

	// AIDEV-NOTE: Perform the scraping
	result, err := h.scraperMgr.ScrapeProfile(ctx, platform, username)
	if err != nil {
		return fmt.Errorf("scraping failed: %w", err)
	}

	if result.Profile == nil {
		return fmt.Errorf("no profile data returned from scraper")
	}

	// AIDEV-NOTE: Convert scrape result to database models
	kol, metrics, content, err := h.scraperMgr.ConvertToModels(platform, result, username)
	if err != nil {
		return fmt.Errorf("failed to convert scrape result: %w", err)
	}

	// AIDEV-NOTE: Store data in database with transaction
	if err := h.storeScrapeResult(ctx, kol, metrics, content); err != nil {
		return fmt.Errorf("failed to store scrape result: %w", err)
	}

	// AIDEV-NOTE: Set job result for API response
	job.Result = map[string]interface{}{
		"kol_id":        kol.ID,
		"platform":      kol.Platform,
		"username":      kol.Username,
		"follower_count": func() int {
			if metrics != nil {
				return metrics.FollowerCount
			}
			return 0
		}(),
		"content_items": len(content),
		"scraped_at":    time.Now().UTC(),
	}

	h.logger.QueueLog(job.ID, job.Type, "processing_completed", logger.Fields{
		"kol_id":        kol.ID,
		"platform":      platform,
		"username":      username,
		"content_items": len(content),
	})

	return nil
}

// storeScrapeResult stores the scraping result in the database
func (h *ScrapeJobHandler) storeScrapeResult(ctx context.Context, kol *models.KOL, metrics *models.KOLMetrics, content []*models.KOLContent) error {
	return h.db.Transaction(ctx, func(tx *database.Tx) error {
		// AIDEV-NOTE: Upsert KOL record
		if err := h.db.UpsertKOL(ctx, kol); err != nil {
			return fmt.Errorf("failed to upsert KOL: %w", err)
		}

		// AIDEV-NOTE: Upsert metrics if available
		if metrics != nil {
			if err := h.db.UpsertKOLMetrics(ctx, metrics); err != nil {
				return fmt.Errorf("failed to upsert KOL metrics: %w", err)
			}
		}

		// AIDEV-NOTE: Bulk upsert content if available
		if len(content) > 0 {
			if err := h.db.BulkUpsertKOLContent(ctx, content); err != nil {
				return fmt.Errorf("failed to upsert KOL content: %w", err)
			}
		}

		return nil
	})
}

// UpdateMetricsJobHandler handles metrics update jobs
type UpdateMetricsJobHandler struct {
	db         *database.DB
	scraperMgr *scrapers.Manager
	logger     *logger.Logger
}

// NewUpdateMetricsJobHandler creates a new metrics update job handler
func NewUpdateMetricsJobHandler(db *database.DB, scraperMgr *scrapers.Manager, logger *logger.Logger) *UpdateMetricsJobHandler {
	return &UpdateMetricsJobHandler{
		db:         db,
		scraperMgr: scraperMgr,
		logger:     logger,
	}
}

// Handle processes a metrics update job
func (h *UpdateMetricsJobHandler) Handle(ctx context.Context, job *queue.Job) error {
	platform := models.Platform(job.Platform)
	username := job.Username

	h.logger.QueueLog(job.ID, job.Type, "metrics_update_started", logger.Fields{
		"platform": platform,
		"username": username,
	})

	// AIDEV-NOTE: Check if force refresh is requested
	forceRefresh := false
	if job.Params != nil {
		if fr, ok := job.Params["force_refresh"].(bool); ok {
			forceRefresh = fr
		}
	}

	// AIDEV-NOTE: Check if metrics are recent enough (unless force refresh)
	if !forceRefresh {
		if recent, err := h.hasRecentMetrics(ctx, platform, username); err == nil && recent {
			h.logger.QueueLog(job.ID, job.Type, "metrics_already_recent", logger.Fields{
				"platform": platform,
				"username": username,
			})
			return nil
		}
	}

	// AIDEV-NOTE: Perform scraping to get updated metrics
	result, err := h.scraperMgr.ScrapeProfile(ctx, platform, username)
	if err != nil {
		return fmt.Errorf("metrics scraping failed: %w", err)
	}

	// AIDEV-NOTE: Convert result to models and update only metrics
	kol, metrics, _, err := h.scraperMgr.ConvertToModels(platform, result, username)
	if err != nil {
		return fmt.Errorf("failed to convert metrics result: %w", err)
	}

	// AIDEV-NOTE: Update only the metrics and basic profile info
	if err := h.updateMetricsData(ctx, kol, metrics); err != nil {
		return fmt.Errorf("failed to update metrics: %w", err)
	}

	job.Result = map[string]interface{}{
		"kol_id":         kol.ID,
		"platform":       kol.Platform,
		"username":       kol.Username,
		"metrics_updated": metrics != nil,
		"updated_at":     time.Now().UTC(),
	}

	h.logger.QueueLog(job.ID, job.Type, "metrics_update_completed", logger.Fields{
		"kol_id":   kol.ID,
		"platform": platform,
		"username": username,
	})

	return nil
}

// hasRecentMetrics checks if KOL has recent metrics (within last 6 hours)
func (h *UpdateMetricsJobHandler) hasRecentMetrics(ctx context.Context, platform models.Platform, username string) (bool, error) {
	query := `
		SELECT COUNT(*) FROM kol_metrics km
		JOIN kols k ON k.id = km.kol_id
		WHERE k.platform = $1 AND k.username = $2 
		AND km.metrics_date >= $3`

	var count int
	err := h.db.GetContext(ctx, &count, query, platform, username, time.Now().Add(-6*time.Hour))
	if err != nil {
		return false, err
	}

	return count > 0, nil
}

// updateMetricsData updates KOL and metrics data
func (h *UpdateMetricsJobHandler) updateMetricsData(ctx context.Context, kol *models.KOL, metrics *models.KOLMetrics) error {
	return h.db.Transaction(ctx, func(tx *database.Tx) error {
		// AIDEV-NOTE: Update basic KOL info (follower count might affect tier)
		if err := h.db.UpsertKOL(ctx, kol); err != nil {
			return fmt.Errorf("failed to update KOL: %w", err)
		}

		// AIDEV-NOTE: Insert new metrics record
		if metrics != nil {
			if err := h.db.UpsertKOLMetrics(ctx, metrics); err != nil {
				return fmt.Errorf("failed to update metrics: %w", err)
			}
		}

		return nil
	})
}

// UpdateProfileJobHandler handles profile update jobs
type UpdateProfileJobHandler struct {
	db         *database.DB
	scraperMgr *scrapers.Manager
	logger     *logger.Logger
}

// NewUpdateProfileJobHandler creates a new profile update job handler
func NewUpdateProfileJobHandler(db *database.DB, scraperMgr *scrapers.Manager, logger *logger.Logger) *UpdateProfileJobHandler {
	return &UpdateProfileJobHandler{
		db:         db,
		scraperMgr: scraperMgr,
		logger:     logger,
	}
}

// Handle processes a profile update job
func (h *UpdateProfileJobHandler) Handle(ctx context.Context, job *queue.Job) error {
	platform := models.Platform(job.Platform)
	username := job.Username

	// AIDEV-NOTE: Get KOL ID from job params if provided
	var kolID string
	if job.Params != nil {
		if id, ok := job.Params["kol_id"].(string); ok {
			kolID = id
		}
	}

	h.logger.QueueLog(job.ID, job.Type, "profile_update_started", logger.Fields{
		"platform": platform,
		"username": username,
		"kol_id":   kolID,
	})

	// AIDEV-NOTE: Perform scraping to get updated profile
	result, err := h.scraperMgr.ScrapeProfile(ctx, platform, username)
	if err != nil {
		return fmt.Errorf("profile scraping failed: %w", err)
	}

	// AIDEV-NOTE: Convert result to models
	kol, metrics, content, err := h.scraperMgr.ConvertToModels(platform, result, username)
	if err != nil {
		return fmt.Errorf("failed to convert profile result: %w", err)
	}

	// AIDEV-NOTE: If we have an existing KOL ID, preserve it
	if kolID != "" {
		kol.ID = kolID
	}

	// AIDEV-NOTE: Update profile, metrics, and recent content
	if err := h.updateProfileData(ctx, kol, metrics, content); err != nil {
		return fmt.Errorf("failed to update profile: %w", err)
	}

	job.Result = map[string]interface{}{
		"kol_id":         kol.ID,
		"platform":       kol.Platform,
		"username":       kol.Username,
		"profile_updated": true,
		"metrics_updated": metrics != nil,
		"content_items":   len(content),
		"updated_at":      time.Now().UTC(),
	}

	h.logger.QueueLog(job.ID, job.Type, "profile_update_completed", logger.Fields{
		"kol_id":        kol.ID,
		"platform":      platform,
		"username":      username,
		"content_items": len(content),
	})

	return nil
}

// updateProfileData updates KOL profile, metrics, and content
func (h *UpdateProfileJobHandler) updateProfileData(ctx context.Context, kol *models.KOL, metrics *models.KOLMetrics, content []*models.KOLContent) error {
	return h.db.Transaction(ctx, func(tx *database.Tx) error {
		// AIDEV-NOTE: Update KOL profile information
		if err := h.db.UpsertKOL(ctx, kol); err != nil {
			return fmt.Errorf("failed to update KOL profile: %w", err)
		}

		// AIDEV-NOTE: Update metrics if available
		if metrics != nil {
			if err := h.db.UpsertKOLMetrics(ctx, metrics); err != nil {
				return fmt.Errorf("failed to update metrics: %w", err)
			}
		}

		// AIDEV-NOTE: Update recent content (limit to most recent items to avoid bloat)
		if len(content) > 0 {
			// Limit to 20 most recent items for profile updates
			maxItems := 20
			if len(content) > maxItems {
				content = content[:maxItems]
			}

			if err := h.db.BulkUpsertKOLContent(ctx, content); err != nil {
				return fmt.Errorf("failed to update content: %w", err)
			}
		}

		return nil
	})
}

// JobHandlerFactory creates job handlers based on job type
type JobHandlerFactory struct {
	db         *database.DB
	scraperMgr *scrapers.Manager
	logger     *logger.Logger
}

// NewJobHandlerFactory creates a new job handler factory
func NewJobHandlerFactory(db *database.DB, scraperMgr *scrapers.Manager, logger *logger.Logger) *JobHandlerFactory {
	return &JobHandlerFactory{
		db:         db,
		scraperMgr: scraperMgr,
		logger:     logger,
	}
}

// RegisterHandlers registers all job handlers with the queue
func (f *JobHandlerFactory) RegisterHandlers(jobQueue *queue.JobQueue) {
	// AIDEV-NOTE: Register scraping job handler
	scrapeHandler := NewScrapeJobHandler(f.db, f.scraperMgr, f.logger)
	jobQueue.RegisterHandler(queue.JobTypeSingleScrape, scrapeHandler)

	// AIDEV-NOTE: Register metrics update handler
	metricsHandler := NewUpdateMetricsJobHandler(f.db, f.scraperMgr, f.logger)
	jobQueue.RegisterHandler(queue.JobTypeUpdateMetrics, metricsHandler)

	// AIDEV-NOTE: Register profile update handler
	profileHandler := NewUpdateProfileJobHandler(f.db, f.scraperMgr, f.logger)
	jobQueue.RegisterHandler(queue.JobTypeUpdateProfile, profileHandler)

	f.logger.Info("All job handlers registered successfully")
}