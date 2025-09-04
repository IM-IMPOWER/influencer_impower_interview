// AIDEV-NOTE: Data query handlers for KOL information retrieval
// Provides endpoints for querying KOL data, statistics, and integration status
package handlers

import (
	"context"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"kol-scraper/internal/models"
	"kol-scraper/pkg/logger"
)

// KOL query parameters
type KOLQueryParams struct {
	Platform         string   `form:"platform"`
	Tier             string   `form:"tier"`
	Category         string   `form:"category"`
	MinFollowers     int      `form:"min_followers"`
	MaxFollowers     int      `form:"max_followers"`
	MinEngagement    float64  `form:"min_engagement"`
	MaxEngagement    float64  `form:"max_engagement"`
	IsVerified       *bool    `form:"is_verified"`
	IsBrandSafe      *bool    `form:"is_brand_safe"`
	Location         string   `form:"location"`
	Languages        []string `form:"languages"`
	LastScrapedDays  int      `form:"last_scraped_days"`
	Limit            int      `form:"limit"`
	Offset           int      `form:"offset"`
	SortBy           string   `form:"sort_by"`
	SortOrder        string   `form:"sort_order"`
}

// GetKOLs retrieves KOLs based on query parameters
func (h *Handler) GetKOLs(c *gin.Context) {
	var params KOLQueryParams
	if err := c.ShouldBindQuery(&params); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Invalid query parameters: " + err.Error(),
		})
		return
	}

	// AIDEV-NOTE: Set default values
	if params.Limit <= 0 || params.Limit > 100 {
		params.Limit = 20
	}
	if params.SortBy == "" {
		params.SortBy = "updated_at"
	}
	if params.SortOrder == "" {
		params.SortOrder = "desc"
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	// AIDEV-NOTE: Build dynamic query based on parameters
	query, args, err := h.buildKOLQuery(params)
	if err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Invalid query parameters: " + err.Error(),
		})
		return
	}

	// AIDEV-NOTE: Execute query
	var kols []models.KOL
	if err := h.db.SelectContext(ctx, &kols, query, args...); err != nil {
		h.logger.DatabaseLog("select", "kols", logger.Fields{
			"error": err.Error(),
			"query": query,
		})
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   "Database query failed",
		})
		return
	}

	// AIDEV-NOTE: Get total count for pagination
	countQuery := h.buildKOLCountQuery(params)
	var totalCount int
	if err := h.db.GetContext(ctx, &totalCount, countQuery, args[:len(args)-2]...); err != nil {
		h.logger.ErrorLog(err, "count_kols", logger.Fields{})
		totalCount = len(kols) // Fallback
	}

	// AIDEV-NOTE: Enrich KOL data with latest metrics
	enrichedKOLs, err := h.enrichKOLsWithMetrics(ctx, kols)
	if err != nil {
		h.logger.ErrorLog(err, "enrich_kols", logger.Fields{})
		// Continue with basic data if enrichment fails
		enrichedKOLs = kols
	}

	h.logger.HTTPLog(c.Request.Method, c.Request.URL.Path, http.StatusOK, logger.Fields{
		"kols_returned": len(enrichedKOLs),
		"total_count":   totalCount,
		"filters_used":  h.getActiveFilters(params),
	})

	response := map[string]interface{}{
		"kols":        enrichedKOLs,
		"total":       totalCount,
		"limit":       params.Limit,
		"offset":      params.Offset,
		"has_more":    params.Offset+len(enrichedKOLs) < totalCount,
	}

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data:    response,
	})
}

// GetKOL retrieves a single KOL by ID with detailed information
func (h *Handler) GetKOL(c *gin.Context) {
	kolID := c.Param("id")
	if kolID == "" {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "KOL ID is required",
		})
		return
	}

	includeMetrics := c.Query("include_metrics") == "true"
	includeContent := c.Query("include_content") == "true"
	includeProfile := c.Query("include_profile") == "true"

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	// AIDEV-NOTE: Get basic KOL information
	var kol models.KOL
	query := `SELECT * FROM kols WHERE id = $1 AND is_active = true`
	if err := h.db.GetContext(ctx, &kol, query, kolID); err != nil {
		statusCode := http.StatusNotFound
		if !strings.Contains(err.Error(), "no rows") {
			statusCode = http.StatusInternalServerError
		}

		c.JSON(statusCode, APIResponse{
			Success: false,
			Error:   "KOL not found",
		})
		return
	}

	response := map[string]interface{}{
		"kol": kol,
	}

	// AIDEV-NOTE: Include metrics if requested
	if includeMetrics {
		metrics, err := h.getKOLMetrics(ctx, kolID)
		if err != nil {
			h.logger.ErrorLog(err, "get_kol_metrics", logger.Fields{
				"kol_id": kolID,
			})
		} else {
			response["metrics"] = metrics
		}
	}

	// AIDEV-NOTE: Include content if requested
	if includeContent {
		content, err := h.getKOLContent(ctx, kolID, 10) // Limit to 10 recent items
		if err != nil {
			h.logger.ErrorLog(err, "get_kol_content", logger.Fields{
				"kol_id": kolID,
			})
		} else {
			response["content"] = content
		}
	}

	// AIDEV-NOTE: Include extended profile if requested
	if includeProfile {
		profiles, err := h.getKOLProfiles(ctx, kolID)
		if err != nil {
			h.logger.ErrorLog(err, "get_kol_profiles", logger.Fields{
				"kol_id": kolID,
			})
		} else {
			response["profiles"] = profiles
		}
	}

	h.logger.HTTPLog(c.Request.Method, c.Request.URL.Path, http.StatusOK, logger.Fields{
		"kol_id":          kolID,
		"include_metrics": includeMetrics,
		"include_content": includeContent,
		"include_profile": includeProfile,
	})

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data:    response,
	})
}

// GetStats returns overall statistics about KOL data
func (h *Handler) GetStats(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Second)
	defer cancel()

	stats := map[string]interface{}{
		"timestamp": time.Now().UTC(),
	}

	// AIDEV-NOTE: Get basic KOL counts
	kolStats, err := h.getKOLStats(ctx)
	if err != nil {
		h.logger.ErrorLog(err, "get_kol_stats", logger.Fields{})
		c.JSON(http.StatusInternalServerError, APIResponse{
			Success: false,
			Error:   "Failed to retrieve statistics",
		})
		return
	}
	stats["kols"] = kolStats

	// AIDEV-NOTE: Get platform distribution
	platformStats, err := h.getPlatformStats(ctx)
	if err != nil {
		h.logger.ErrorLog(err, "get_platform_stats", logger.Fields{})
	} else {
		stats["platforms"] = platformStats
	}

	// AIDEV-NOTE: Get tier distribution
	tierStats, err := h.getTierStats(ctx)
	if err != nil {
		h.logger.ErrorLog(err, "get_tier_stats", logger.Fields{})
	} else {
		stats["tiers"] = tierStats
	}

	// AIDEV-NOTE: Get category distribution
	categoryStats, err := h.getCategoryStats(ctx)
	if err != nil {
		h.logger.ErrorLog(err, "get_category_stats", logger.Fields{})
	} else {
		stats["categories"] = categoryStats
	}

	// AIDEV-NOTE: Get scraping statistics
	if h.queue != nil {
		queueStats, err := h.queue.GetQueueStats(ctx)
		if err != nil {
			h.logger.ErrorLog(err, "get_queue_stats", logger.Fields{})
		} else {
			stats["scraping"] = queueStats
		}
	}

	c.JSON(http.StatusOK, APIResponse{
		Success: true,
		Data:    stats,
	})
}

// AIDEV-NOTE: Integration endpoints

// WebhookCallback handles webhook callbacks from external services
func (h *Handler) WebhookCallback(c *gin.Context) {
	// AIDEV-NOTE: Verify webhook signature if configured
	signature := c.GetHeader("X-Webhook-Signature")
	if signature == "" {
		c.JSON(http.StatusUnauthorized, APIResponse{
			Success: false,
			Error:   "Missing webhook signature",
		})
		return
	}

	var payload map[string]interface{}
	if err := c.ShouldBindJSON(&payload); err != nil {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Invalid webhook payload",
		})
		return
	}

	// AIDEV-NOTE: Process webhook based on type
	webhookType, ok := payload["type"].(string)
	if !ok {
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Missing webhook type",
		})
		return
	}

	h.logger.IntegrationLog("webhook", "received", logger.Fields{
		"type":    webhookType,
		"payload": payload,
	})

	switch webhookType {
	case "scrape_completed":
		h.handleScrapeCompletedWebhook(c, payload)
	case "metrics_updated":
		h.handleMetricsUpdatedWebhook(c, payload)
	case "profile_updated":
		h.handleProfileUpdatedWebhook(c, payload)
	default:
		h.logger.IntegrationLog("webhook", "unknown_type", logger.Fields{
			"type": webhookType,
		})
		c.JSON(http.StatusBadRequest, APIResponse{
			Success: false,
			Error:   "Unknown webhook type: " + webhookType,
		})
		return
	}
}

// IntegrationStatus returns the status of integrations with external services
func (h *Handler) IntegrationStatus(c *gin.Context) {
	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	status := map[string]interface{}{
		"timestamp": time.Now().UTC(),
		"services":  make(map[string]interface{}),
	}

	// AIDEV-NOTE: Check database connectivity
	dbHealthy := true
	if err := h.db.Health(ctx); err != nil {
		dbHealthy = false
	}
	status["services"].(map[string]interface{})["database"] = map[string]interface{}{
		"status":  getHealthStatus(dbHealthy),
		"healthy": dbHealthy,
	}

	// AIDEV-NOTE: Check queue connectivity
	queueHealthy := true
	if _, err := h.queue.GetQueueStats(ctx); err != nil {
		queueHealthy = false
	}
	status["services"].(map[string]interface{})["queue"] = map[string]interface{}{
		"status":  getHealthStatus(queueHealthy),
		"healthy": queueHealthy,
	}

	// AIDEV-NOTE: Check scraper status
	scraperStatus := make(map[string]bool)
	if h.scraperMgr != nil {
		scraperStatus = h.scraperMgr.HealthCheck(ctx)
	}
	status["services"].(map[string]interface{})["scrapers"] = scraperStatus

	// AIDEV-NOTE: Overall integration status
	overallHealthy := dbHealthy && queueHealthy && len(scraperStatus) > 0
	status["overall_status"] = getHealthStatus(overallHealthy)
	status["healthy"] = overallHealthy

	statusCode := http.StatusOK
	if !overallHealthy {
		statusCode = http.StatusServiceUnavailable
	}

	c.JSON(statusCode, APIResponse{
		Success: overallHealthy,
		Data:    status,
	})
}

// AIDEV-NOTE: Helper functions for data queries

func (h *Handler) buildKOLQuery(params KOLQueryParams) (string, []interface{}, error) {
	baseQuery := `
		SELECT k.*, 
		       COALESCE(m.follower_count, 0) as follower_count,
		       COALESCE(m.engagement_rate, 0) as engagement_rate
		FROM kols k
		LEFT JOIN LATERAL (
			SELECT follower_count, engagement_rate
			FROM kol_metrics km
			WHERE km.kol_id = k.id
			ORDER BY km.metrics_date DESC
			LIMIT 1
		) m ON true
		WHERE k.is_active = true`

	var conditions []string
	var args []interface{}
	argIndex := 1

	// AIDEV-NOTE: Add filters based on parameters
	if params.Platform != "" {
		conditions = append(conditions, "k.platform = $"+strconv.Itoa(argIndex))
		args = append(args, params.Platform)
		argIndex++
	}

	if params.Tier != "" {
		conditions = append(conditions, "k.tier = $"+strconv.Itoa(argIndex))
		args = append(args, params.Tier)
		argIndex++
	}

	if params.Category != "" {
		conditions = append(conditions, "k.primary_category = $"+strconv.Itoa(argIndex))
		args = append(args, params.Category)
		argIndex++
	}

	if params.MinFollowers > 0 {
		conditions = append(conditions, "COALESCE(m.follower_count, 0) >= $"+strconv.Itoa(argIndex))
		args = append(args, params.MinFollowers)
		argIndex++
	}

	if params.MaxFollowers > 0 {
		conditions = append(conditions, "COALESCE(m.follower_count, 0) <= $"+strconv.Itoa(argIndex))
		args = append(args, params.MaxFollowers)
		argIndex++
	}

	if params.MinEngagement > 0 {
		conditions = append(conditions, "COALESCE(m.engagement_rate, 0) >= $"+strconv.Itoa(argIndex))
		args = append(args, params.MinEngagement)
		argIndex++
	}

	if params.MaxEngagement > 0 {
		conditions = append(conditions, "COALESCE(m.engagement_rate, 0) <= $"+strconv.Itoa(argIndex))
		args = append(args, params.MaxEngagement)
		argIndex++
	}

	if params.IsVerified != nil {
		conditions = append(conditions, "k.is_verified = $"+strconv.Itoa(argIndex))
		args = append(args, *params.IsVerified)
		argIndex++
	}

	if params.IsBrandSafe != nil {
		conditions = append(conditions, "k.is_brand_safe = $"+strconv.Itoa(argIndex))
		args = append(args, *params.IsBrandSafe)
		argIndex++
	}

	if params.Location != "" {
		conditions = append(conditions, "k.location ILIKE $"+strconv.Itoa(argIndex))
		args = append(args, "%"+params.Location+"%")
		argIndex++
	}

	if len(params.Languages) > 0 {
		conditions = append(conditions, "k.languages && $"+strconv.Itoa(argIndex))
		args = append(args, params.Languages)
		argIndex++
	}

	if params.LastScrapedDays > 0 {
		conditions = append(conditions, "k.last_scraped >= $"+strconv.Itoa(argIndex))
		args = append(args, time.Now().AddDate(0, 0, -params.LastScrapedDays))
		argIndex++
	}

	// AIDEV-NOTE: Add conditions to query
	if len(conditions) > 0 {
		baseQuery += " AND " + strings.Join(conditions, " AND ")
	}

	// AIDEV-NOTE: Add sorting
	orderBy := "k." + params.SortBy
	if params.SortBy == "follower_count" || params.SortBy == "engagement_rate" {
		orderBy = "m." + params.SortBy
	}

	baseQuery += " ORDER BY " + orderBy
	if strings.ToLower(params.SortOrder) == "desc" {
		baseQuery += " DESC"
	}

	// AIDEV-NOTE: Add pagination
	baseQuery += " LIMIT $" + strconv.Itoa(argIndex) + " OFFSET $" + strconv.Itoa(argIndex+1)
	args = append(args, params.Limit, params.Offset)

	return baseQuery, args, nil
}

func (h *Handler) buildKOLCountQuery(params KOLQueryParams) string {
	return strings.Replace(
		strings.Split(h.buildKOLQueryWithoutPagination(params), "ORDER BY")[0],
		"SELECT k.*, COALESCE(m.follower_count, 0) as follower_count, COALESCE(m.engagement_rate, 0) as engagement_rate",
		"SELECT COUNT(*)",
		1,
	)
}

func (h *Handler) buildKOLQueryWithoutPagination(params KOLQueryParams) string {
	query, _, _ := h.buildKOLQuery(params)
	parts := strings.Split(query, " LIMIT ")
	return parts[0]
}

func (h *Handler) getActiveFilters(params KOLQueryParams) map[string]interface{} {
	filters := make(map[string]interface{})

	if params.Platform != "" {
		filters["platform"] = params.Platform
	}
	if params.Tier != "" {
		filters["tier"] = params.Tier
	}
	if params.Category != "" {
		filters["category"] = params.Category
	}
	if params.MinFollowers > 0 {
		filters["min_followers"] = params.MinFollowers
	}
	if params.MaxFollowers > 0 {
		filters["max_followers"] = params.MaxFollowers
	}

	return filters
}

func getHealthStatus(healthy bool) string {
	if healthy {
		return "healthy"
	}
	return "unhealthy"
}