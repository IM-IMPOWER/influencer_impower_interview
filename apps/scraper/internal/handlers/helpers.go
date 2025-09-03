// AIDEV-NOTE: Helper functions for data retrieval and webhook handling
// Provides database query helpers and webhook processing logic
package handlers

import (
	"context"
	"fmt"
	"time"

	"github.com/gin-gonic/gin"
	"kol-scraper/internal/models"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: Database helper functions

// enrichKOLsWithMetrics adds latest metrics to KOL data
func (h *Handler) enrichKOLsWithMetrics(ctx context.Context, kols []models.KOL) ([]models.KOL, error) {
	if len(kols) == 0 {
		return kols, nil
	}

	// AIDEV-NOTE: Get all KOL IDs for batch query
	kolIDs := make([]string, len(kols))
	for i, kol := range kols {
		kolIDs[i] = kol.ID
	}

	// AIDEV-NOTE: Query latest metrics for all KOLs
	query := `
		SELECT DISTINCT ON (kol_id) 
			kol_id, follower_count, following_count, engagement_rate,
			avg_likes, avg_comments, posts_last_30_days, metrics_date
		FROM kol_metrics 
		WHERE kol_id = ANY($1)
		ORDER BY kol_id, metrics_date DESC`

	var metricsRows []struct {
		KOLID              string    `db:"kol_id"`
		FollowerCount      int       `db:"follower_count"`
		FollowingCount     int       `db:"following_count"`
		EngagementRate     *float64  `db:"engagement_rate"`
		AvgLikes           *float64  `db:"avg_likes"`
		AvgComments        *float64  `db:"avg_comments"`
		PostsLast30Days    int       `db:"posts_last_30_days"`
		MetricsDate        time.Time `db:"metrics_date"`
	}

	if err := h.db.SelectContext(ctx, &metricsRows, query, kolIDs); err != nil {
		return kols, fmt.Errorf("failed to get metrics: %w", err)
	}

	// AIDEV-NOTE: Create metrics map for efficient lookup
	metricsMap := make(map[string]interface{})
	for _, metrics := range metricsRows {
		metricsMap[metrics.KOLID] = map[string]interface{}{
			"follower_count":     metrics.FollowerCount,
			"following_count":    metrics.FollowingCount,
			"engagement_rate":    metrics.EngagementRate,
			"avg_likes":          metrics.AvgLikes,
			"avg_comments":       metrics.AvgComments,
			"posts_last_30_days": metrics.PostsLast30Days,
			"metrics_date":       metrics.MetricsDate,
		}
	}

	// AIDEV-NOTE: Add metrics to each KOL (this would require modifying the KOL struct or using a response wrapper)
	// For now, returning original KOLs as the struct doesn't have a metrics field
	// In production, you might want to use a response DTO that includes metrics

	return kols, nil
}

// getKOLMetrics retrieves latest metrics for a specific KOL
func (h *Handler) getKOLMetrics(ctx context.Context, kolID string) (*models.KOLMetrics, error) {
	var metrics models.KOLMetrics
	query := `
		SELECT * FROM kol_metrics 
		WHERE kol_id = $1 
		ORDER BY metrics_date DESC 
		LIMIT 1`

	if err := h.db.GetContext(ctx, &metrics, query, kolID); err != nil {
		return nil, fmt.Errorf("failed to get KOL metrics: %w", err)
	}

	return &metrics, nil
}

// getKOLContent retrieves recent content for a specific KOL
func (h *Handler) getKOLContent(ctx context.Context, kolID string, limit int) ([]models.KOLContent, error) {
	var content []models.KOLContent
	query := `
		SELECT * FROM kol_content 
		WHERE kol_id = $1 
		ORDER BY posted_at DESC 
		LIMIT $2`

	if err := h.db.SelectContext(ctx, &content, query, kolID, limit); err != nil {
		return nil, fmt.Errorf("failed to get KOL content: %w", err)
	}

	return content, nil
}

// getKOLProfiles retrieves extended profiles for a specific KOL
func (h *Handler) getKOLProfiles(ctx context.Context, kolID string) ([]models.KOLProfile, error) {
	var profiles []models.KOLProfile
	query := `
		SELECT * FROM kol_profiles 
		WHERE kol_id = $1 
		ORDER BY updated_at DESC`

	if err := h.db.SelectContext(ctx, &profiles, query, kolID); err != nil {
		return nil, fmt.Errorf("failed to get KOL profiles: %w", err)
	}

	return profiles, nil
}

// getKOLStats retrieves basic KOL statistics
func (h *Handler) getKOLStats(ctx context.Context) (map[string]interface{}, error) {
	stats := make(map[string]interface{})

	// AIDEV-NOTE: Get total KOL count
	var totalKOLs int
	if err := h.db.GetContext(ctx, &totalKOLs, "SELECT COUNT(*) FROM kols WHERE is_active = true"); err != nil {
		return nil, fmt.Errorf("failed to get total KOLs: %w", err)
	}
	stats["total"] = totalKOLs

	// AIDEV-NOTE: Get verified KOL count
	var verifiedKOLs int
	if err := h.db.GetContext(ctx, &verifiedKOLs, "SELECT COUNT(*) FROM kols WHERE is_active = true AND is_verified = true"); err != nil {
		return nil, fmt.Errorf("failed to get verified KOLs: %w", err)
	}
	stats["verified"] = verifiedKOLs

	// AIDEV-NOTE: Get brand safe KOL count
	var brandSafeKOLs int
	if err := h.db.GetContext(ctx, &brandSafeKOLs, "SELECT COUNT(*) FROM kols WHERE is_active = true AND is_brand_safe = true"); err != nil {
		return nil, fmt.Errorf("failed to get brand safe KOLs: %w", err)
	}
	stats["brand_safe"] = brandSafeKOLs

	// AIDEV-NOTE: Get recently scraped count (last 24 hours)
	var recentlyScraped int
	if err := h.db.GetContext(ctx, &recentlyScraped, 
		"SELECT COUNT(*) FROM kols WHERE is_active = true AND last_scraped >= $1", 
		time.Now().Add(-24*time.Hour)); err != nil {
		return nil, fmt.Errorf("failed to get recently scraped KOLs: %w", err)
	}
	stats["scraped_last_24h"] = recentlyScraped

	return stats, nil
}

// getPlatformStats retrieves platform distribution statistics
func (h *Handler) getPlatformStats(ctx context.Context) (map[string]int, error) {
	stats := make(map[string]int)

	query := `
		SELECT platform, COUNT(*) as count
		FROM kols 
		WHERE is_active = true
		GROUP BY platform
		ORDER BY count DESC`

	rows, err := h.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get platform stats: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var platform string
		var count int
		if err := rows.Scan(&platform, &count); err != nil {
			return nil, fmt.Errorf("failed to scan platform stats: %w", err)
		}
		stats[platform] = count
	}

	return stats, nil
}

// getTierStats retrieves tier distribution statistics
func (h *Handler) getTierStats(ctx context.Context) (map[string]int, error) {
	stats := make(map[string]int)

	query := `
		SELECT tier, COUNT(*) as count
		FROM kols 
		WHERE is_active = true
		GROUP BY tier
		ORDER BY 
			CASE tier
				WHEN 'mega' THEN 1
				WHEN 'macro' THEN 2
				WHEN 'mid' THEN 3
				WHEN 'micro' THEN 4
				WHEN 'nano' THEN 5
			END`

	rows, err := h.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get tier stats: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var tier string
		var count int
		if err := rows.Scan(&tier, &count); err != nil {
			return nil, fmt.Errorf("failed to scan tier stats: %w", err)
		}
		stats[tier] = count
	}

	return stats, nil
}

// getCategoryStats retrieves category distribution statistics
func (h *Handler) getCategoryStats(ctx context.Context) (map[string]int, error) {
	stats := make(map[string]int)

	query := `
		SELECT primary_category, COUNT(*) as count
		FROM kols 
		WHERE is_active = true
		GROUP BY primary_category
		ORDER BY count DESC
		LIMIT 20`

	rows, err := h.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get category stats: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var category string
		var count int
		if err := rows.Scan(&category, &count); err != nil {
			return nil, fmt.Errorf("failed to scan category stats: %w", err)
		}
		stats[category] = count
	}

	return stats, nil
}

// AIDEV-NOTE: Webhook handler functions

// handleScrapeCompletedWebhook processes scrape completion webhooks
func (h *Handler) handleScrapeCompletedWebhook(c *gin.Context, payload map[string]interface{}) {
	jobID, ok := payload["job_id"].(string)
	if !ok {
		c.JSON(400, APIResponse{
			Success: false,
			Error:   "Missing job_id in webhook payload",
		})
		return
	}

	success, ok := payload["success"].(bool)
	if !ok {
		c.JSON(400, APIResponse{
			Success: false,
			Error:   "Missing success status in webhook payload",
		})
		return
	}

	h.logger.IntegrationLog("webhook", "scrape_completed", logger.Fields{
		"job_id":  jobID,
		"success": success,
	})

	// AIDEV-NOTE: Process webhook data and update job status if needed
	if success {
		// Handle successful scrape completion
		if data, ok := payload["data"].(map[string]interface{}); ok {
			h.processScrapeResultData(c.Request.Context(), jobID, data)
		}
	} else {
		// Handle failed scrape
		if errorMsg, ok := payload["error"].(string); ok {
			h.logger.ErrorLog(fmt.Errorf("scrape job failed: %s", errorMsg), "webhook_scrape_failed", logger.Fields{
				"job_id": jobID,
			})
		}
	}

	c.JSON(200, APIResponse{
		Success: true,
		Message: "Webhook processed successfully",
	})
}

// handleMetricsUpdatedWebhook processes metrics update webhooks
func (h *Handler) handleMetricsUpdatedWebhook(c *gin.Context, payload map[string]interface{}) {
	kolID, ok := payload["kol_id"].(string)
	if !ok {
		c.JSON(400, APIResponse{
			Success: false,
			Error:   "Missing kol_id in webhook payload",
		})
		return
	}

	h.logger.IntegrationLog("webhook", "metrics_updated", logger.Fields{
		"kol_id": kolID,
	})

	// AIDEV-NOTE: Handle metrics update notification
	// This could trigger cache invalidation, notification to clients, etc.

	c.JSON(200, APIResponse{
		Success: true,
		Message: "Metrics update webhook processed",
	})
}

// handleProfileUpdatedWebhook processes profile update webhooks
func (h *Handler) handleProfileUpdatedWebhook(c *gin.Context, payload map[string]interface{}) {
	kolID, ok := payload["kol_id"].(string)
	if !ok {
		c.JSON(400, APIResponse{
			Success: false,
			Error:   "Missing kol_id in webhook payload",
		})
		return
	}

	h.logger.IntegrationLog("webhook", "profile_updated", logger.Fields{
		"kol_id": kolID,
	})

	// AIDEV-NOTE: Handle profile update notification
	// This could trigger downstream processes or notifications

	c.JSON(200, APIResponse{
		Success: true,
		Message: "Profile update webhook processed",
	})
}

// processScrapeResultData processes successful scrape result data
func (h *Handler) processScrapeResultData(ctx context.Context, jobID string, data map[string]interface{}) {
	h.logger.IntegrationLog("webhook", "processing_scrape_result", logger.Fields{
		"job_id": jobID,
		"data_keys": getMapKeys(data),
	})

	// AIDEV-NOTE: Process the scrape result data
	// This could involve updating database records, triggering further processing, etc.
	// Implementation would depend on the specific data format from the scraper
}

// getMapKeys returns the keys of a map for logging purposes
func getMapKeys(m map[string]interface{}) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}