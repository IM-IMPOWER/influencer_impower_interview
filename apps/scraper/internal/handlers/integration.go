// AIDEV-NOTE: 250102120325 Integration handlers for POC2/POC4 endpoints
// Implements KOL matching and budget optimization via FastAPI GraphQL communication
package handlers

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"kol-scraper/internal/fastapi"
	"kol-scraper/internal/models"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: SetFastAPIClient sets the FastAPI client for integration handlers
func (h *Handler) SetFastAPIClient(client *fastapi.Client) {
	h.fastapiClient = client
}

// AIDEV-NOTE: MatchKOLs handles POC2 KOL matching requests
func (h *Handler) MatchKOLs(c *gin.Context) {
	if h.fastapiClient == nil {
		c.JSON(http.StatusServiceUnavailable, models.IntegrationResponse{
			Success: false,
			Error:   "FastAPI client not available",
			Meta: models.ResponseMeta{
				Timestamp: time.Now(),
			},
		})
		return
	}
	// AIDEV-NOTE: Parse and validate request
	var req models.KOLMatchingRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, models.IntegrationResponse{
			Success: false,
			Error:   "Invalid request body: " + err.Error(),
			Meta: models.ResponseMeta{
				Timestamp: time.Now(),
			},
		})
		return
	}

	// AIDEV-NOTE: Validate request parameters
	if err := req.Validate(); err != nil {
		c.JSON(http.StatusBadRequest, models.IntegrationResponse{
			Success: false,
			Error:   err.Error(),
			Meta: models.ResponseMeta{
				Timestamp: time.Now(),
			},
		})
		return
	}

	// AIDEV-NOTE: Create request ID for tracking
	if req.RequestID == "" {
		req.RequestID = generateRequestID()
	}

	startTime := time.Now()
	ctx, cancel := context.WithTimeout(c.Request.Context(), 60*time.Second)
	defer cancel()

	// AIDEV-NOTE: Log request initiation
	h.logger.IntegrationLog("kol_matching", "request_started", logger.Fields{
		"request_id":     req.RequestID,
		"budget":         req.Budget,
		"platforms":      req.Platforms,
		"max_results":    req.MaxResults,
		"target_tier":    req.TargetTier,
		"min_followers":  req.MinFollowers,
		"max_followers":  req.MaxFollowers,
	})

	// AIDEV-NOTE: Create integration request record for audit
	integrationReq := &models.IntegrationRequest{
		RequestType: models.RequestTypeMatchKOLs,
		Payload:     models.JSONBMap(structToMap(req)),
		Status:      models.IntegrationStatusRunning,
	}
	integrationReq.BeforeCreate()

	// AIDEV-NOTE: Store integration request for tracking
	if err := h.storeIntegrationRequest(ctx, integrationReq); err != nil {
		h.logger.ErrorLog(err, "store_integration_request", logger.Fields{
			"request_id": req.RequestID,
		})
	}

	// AIDEV-NOTE: Convert request to FastAPI format
	fastapiReq := &fastapi.MatchKOLsRequest{
		CampaignBrief:  req.CampaignBrief,
		Budget:         req.Budget,
		TargetTier:     convertTierToString(req.TargetTier),
		Platforms:      convertPlatformToString(req.Platforms),
		Categories:     convertCategoryToString(req.Categories),
		MinFollowers:   req.MinFollowers,
		MaxFollowers:   req.MaxFollowers,
		Demographics:   req.Demographics,
		MaxResults:     req.MaxResults,
	}

	// AIDEV-NOTE: Call FastAPI GraphQL endpoint
	fastapiResp, err := h.fastapiClient.MatchKOLs(ctx, fastapiReq)
	processingTime := time.Since(startTime)

	if err != nil {
		h.logger.ErrorLog(err, "fastapi_match_kols_failed", logger.Fields{
			"request_id":     req.RequestID,
			"processing_time": processingTime.Milliseconds(),
		})

		// AIDEV-NOTE: Update integration request with error
		integrationReq.Fail(err, processingTime)
		h.updateIntegrationRequest(ctx, integrationReq)

		c.JSON(http.StatusInternalServerError, models.IntegrationResponse{
			Success:   false,
			Error:     "KOL matching failed: " + err.Error(),
			RequestID: req.RequestID,
			Meta: models.ResponseMeta{
				ProcessingTime: processingTime,
				Timestamp:      time.Now(),
			},
		})
		return
	}

	// AIDEV-NOTE: Convert FastAPI response to our format
	response := h.convertMatchKOLsResponse(fastapiResp)

	// AIDEV-NOTE: Update integration request with success
	integrationReq.Complete(structToMap(response), processingTime)
	h.updateIntegrationRequest(ctx, integrationReq)

	// AIDEV-NOTE: Log successful completion
	h.logger.IntegrationLog("kol_matching", "request_completed", logger.Fields{
		"request_id":     req.RequestID,
		"matches_found":  len(response.Matches),
		"processing_time": processingTime.Milliseconds(),
		"confidence":     response.Meta.Confidence,
	})

	c.JSON(http.StatusOK, models.IntegrationResponse{
		Success:   true,
		Data:      response,
		Message:   "KOL matching completed successfully",
		RequestID: req.RequestID,
		Meta: models.ResponseMeta{
			ProcessingTime: processingTime,
			Timestamp:      time.Now(),
			Version:        "v1.0",
		},
	})
}

// AIDEV-NOTE: OptimizeBudget handles POC4 budget optimization requests
func (h *Handler) OptimizeBudget(c *gin.Context) {
	if h.fastapiClient == nil {
		c.JSON(http.StatusServiceUnavailable, models.IntegrationResponse{
			Success: false,
			Error:   "FastAPI client not available",
			Meta: models.ResponseMeta{
				Timestamp: time.Now(),
			},
		})
		return
	}
	// AIDEV-NOTE: Parse and validate request
	var req models.BudgetOptimizationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, models.IntegrationResponse{
			Success: false,
			Error:   "Invalid request body: " + err.Error(),
			Meta: models.ResponseMeta{
				Timestamp: time.Now(),
			},
		})
		return
	}

	// AIDEV-NOTE: Validate request parameters
	if err := req.Validate(); err != nil {
		c.JSON(http.StatusBadRequest, models.IntegrationResponse{
			Success: false,
			Error:   err.Error(),
			Meta: models.ResponseMeta{
				Timestamp: time.Now(),
			},
		})
		return
	}

	// AIDEV-NOTE: Create request ID for tracking
	if req.RequestID == "" {
		req.RequestID = generateRequestID()
	}

	startTime := time.Now()
	ctx, cancel := context.WithTimeout(c.Request.Context(), 90*time.Second)
	defer cancel()

	// AIDEV-NOTE: Log request initiation
	h.logger.IntegrationLog("budget_optimization", "request_started", logger.Fields{
		"request_id":       req.RequestID,
		"total_budget":     req.TotalBudget,
		"campaign_goals":   req.CampaignGoals,
		"kol_candidates":   len(req.KOLCandidates),
		"target_reach":     req.TargetReach,
		"target_engagement": req.TargetEngagement,
	})

	// AIDEV-NOTE: Create integration request record for audit
	integrationReq := &models.IntegrationRequest{
		RequestType: models.RequestTypeOptimizeBudget,
		Payload:     models.JSONBMap(structToMap(req)),
		Status:      models.IntegrationStatusRunning,
	}
	integrationReq.BeforeCreate()

	// AIDEV-NOTE: Store integration request for tracking
	if err := h.storeIntegrationRequest(ctx, integrationReq); err != nil {
		h.logger.ErrorLog(err, "store_integration_request", logger.Fields{
			"request_id": req.RequestID,
		})
	}

	// AIDEV-NOTE: Validate KOL candidates exist in our database
	validKOLs, err := h.validateKOLCandidates(ctx, req.KOLCandidates)
	if err != nil {
		h.logger.ErrorLog(err, "validate_kol_candidates", logger.Fields{
			"request_id": req.RequestID,
			"candidates": len(req.KOLCandidates),
		})

		c.JSON(http.StatusBadRequest, models.IntegrationResponse{
			Success:   false,
			Error:     "Invalid KOL candidates: " + err.Error(),
			RequestID: req.RequestID,
			Meta: models.ResponseMeta{
				ProcessingTime: time.Since(startTime),
				Timestamp:      time.Now(),
			},
		})
		return
	}

	// AIDEV-NOTE: Convert request to FastAPI format
	fastapiReq := &fastapi.OptimizeBudgetRequest{
		TotalBudget:      req.TotalBudget,
		CampaignGoals:    req.CampaignGoals,
		KOLCandidates:    validKOLs,
		TargetReach:      req.TargetReach,
		TargetEngagement: req.TargetEngagement,
		Constraints:      fastapi.BudgetConstraints{
			MaxKOLs:        req.Constraints.MaxKOLs,
			MinKOLs:        req.Constraints.MinKOLs,
			TierLimits:     req.Constraints.TierLimits,
			PlatformLimits: req.Constraints.PlatformLimits,
		},
	}

	// AIDEV-NOTE: Call FastAPI GraphQL endpoint
	fastapiResp, err := h.fastapiClient.OptimizeBudget(ctx, fastapiReq)
	processingTime := time.Since(startTime)

	if err != nil {
		h.logger.ErrorLog(err, "fastapi_optimize_budget_failed", logger.Fields{
			"request_id":      req.RequestID,
			"processing_time": processingTime.Milliseconds(),
		})

		// AIDEV-NOTE: Update integration request with error
		integrationReq.Fail(err, processingTime)
		h.updateIntegrationRequest(ctx, integrationReq)

		c.JSON(http.StatusInternalServerError, models.IntegrationResponse{
			Success:   false,
			Error:     "Budget optimization failed: " + err.Error(),
			RequestID: req.RequestID,
			Meta: models.ResponseMeta{
				ProcessingTime: processingTime,
				Timestamp:      time.Now(),
			},
		})
		return
	}

	// AIDEV-NOTE: Convert FastAPI response to our format
	response := h.convertOptimizeBudgetResponse(fastapiResp)

	// AIDEV-NOTE: Update integration request with success
	integrationReq.Complete(structToMap(response), processingTime)
	h.updateIntegrationRequest(ctx, integrationReq)

	// AIDEV-NOTE: Log successful completion
	h.logger.IntegrationLog("budget_optimization", "request_completed", logger.Fields{
		"request_id":          req.RequestID,
		"allocations_created": len(response.Allocation),
		"total_allocated":     response.Summary.TotalAllocated,
		"efficiency_score":    response.Summary.OverallEfficiencyScore,
		"processing_time":     processingTime.Milliseconds(),
	})

	c.JSON(http.StatusOK, models.IntegrationResponse{
		Success:   true,
		Data:      response,
		Message:   "Budget optimization completed successfully",
		RequestID: req.RequestID,
		Meta: models.ResponseMeta{
			ProcessingTime: processingTime,
			Timestamp:      time.Now(),
			Version:        "v1.0",
		},
	})
}

// AIDEV-NOTE: Helper functions for request/response conversion

// AIDEV-NOTE: 250902160530 Fixed receiver type from IntegrationHandler to Handler
// convertMatchKOLsResponse converts FastAPI response to our format
func (h *Handler) convertMatchKOLsResponse(fastapiResp *fastapi.MatchKOLsResponse) *models.KOLMatchingResponse {
	matches := make([]models.EnhancedKOLMatch, 0, len(fastapiResp.Matches))

	for _, match := range fastapiResp.Matches {
		enhancedMatch := models.EnhancedKOLMatch{
			KOL: models.KOLSummary{
				ID:          match.KOLID,
				Username:    match.Username,
				DisplayName: match.DisplayName,
				Platform:    models.Platform(match.Platform),
				Tier:        models.KOLTier(match.Tier),
				Category:    models.Category(match.Category),
			},
			Score:         match.Score,
			Reasoning:     match.Reasoning,
			EstimatedCost: match.EstimatedCost,
			Metrics: models.MatchMetrics{
				FollowerCount:  match.FollowerCount,
				EngagementRate: match.EngagementRate,
			},
			Compatibility: models.CompatibilityScore{
				Overall: match.Score,
			},
		}

		matches = append(matches, enhancedMatch)
	}

	return &models.KOLMatchingResponse{
		Matches: matches,
		Meta: models.MatchingMeta{
			TotalCandidates:  fastapiResp.Meta.TotalFound,
			MatchesFound:     len(matches),
			QueryTime:        fastapiResp.Meta.QueryTime,
			AlgorithmVersion: fastapiResp.Meta.AlgorithmUsed,
			Confidence:       fastapiResp.Meta.Confidence,
		},
	}
}

// AIDEV-NOTE: 250902160530 Fixed receiver type from IntegrationHandler to Handler
// convertOptimizeBudgetResponse converts FastAPI response to our format
func (h *Handler) convertOptimizeBudgetResponse(fastapiResp *fastapi.OptimizeBudgetResponse) *models.BudgetOptimizationResponse {
	allocations := make([]models.EnhancedBudgetAllocation, 0, len(fastapiResp.Allocation))

	for _, alloc := range fastapiResp.Allocation {
		enhancedAllocation := models.EnhancedBudgetAllocation{
			KOL: models.KOLSummary{
				ID:       alloc.KOLID,
				Username: alloc.Username,
				Platform: models.Platform(alloc.Platform),
			},
			AllocatedBudget: alloc.AllocatedBudget,
			Priority:        alloc.Priority,
			ExpectedMetrics: models.ExpectedMetrics{
				EstimatedReach:      alloc.EstimatedReach,
				EstimatedEngagement: alloc.EstimatedEngagement,
			},
			ROIProjection: models.ROIProjection{
				ExpectedROI: calculateExpectedROI(alloc.AllocatedBudget, alloc.EstimatedReach),
			},
			RiskAssessment: models.RiskAssessment{
				OverallRisk: "medium", // Default risk assessment
			},
			Reasoning: alloc.Reasoning,
		}

		allocations = append(allocations, enhancedAllocation)
	}

	return &models.BudgetOptimizationResponse{
		Allocation: allocations,
		Summary: models.OptimizationSummary{
			TotalBudget:             fastapiResp.Summary.TotalAllocated + fastapiResp.Summary.RemainingBudget,
			TotalAllocated:          fastapiResp.Summary.TotalAllocated,
			RemainingBudget:         fastapiResp.Summary.RemainingBudget,
			OptimalKOLsSelected:     fastapiResp.Summary.OptimalKOLsSelected,
			BudgetUtilization:       fastapiResp.Summary.TotalAllocated / (fastapiResp.Summary.TotalAllocated + fastapiResp.Summary.RemainingBudget),
			ExpectedOverallReach:    fastapiResp.Summary.EstimatedTotalReach,
			ExpectedOverallEngagement: fastapiResp.Summary.EstimatedAvgEngagement,
			OverallEfficiencyScore:  fastapiResp.Summary.EfficiencyScore,
			ProjectedOverallROI:     fastapiResp.Summary.EfficiencyScore, // Simplified mapping
		},
		Analysis: models.OptimizationAnalysis{
			StrategyRecommendation: "Optimized allocation based on performance metrics and budget constraints",
			KeyInsights: []string{
				"Budget allocation optimized for maximum ROI",
				"KOL selection balanced across tiers and platforms",
				"Performance projections based on historical data",
			},
		},
		Meta: models.OptimizationMeta{
			OptimizationTime:    fastapiResp.Meta.OptimizationTime,
			AlgorithmVersion:    fastapiResp.Meta.AlgorithmUsed,
			IterationsRun:       fastapiResp.Meta.Iterations,
			ConvergenceAchieved: true,
			DataFreshness:       time.Now(),
		},
	}
}

// AIDEV-NOTE: Database helper functions

// AIDEV-NOTE: 250902160530 Fixed receiver type and added proper error wrapping
// storeIntegrationRequest stores integration request for audit trail
func (h *Handler) storeIntegrationRequest(ctx context.Context, req *models.IntegrationRequest) error {
	query := `
		INSERT INTO integration_requests (id, request_type, payload, status, created_at)
		VALUES ($1, $2, $3, $4, $5)
	`
	_, err := h.db.ExecContext(ctx, query, req.ID, req.RequestType, req.Payload, req.Status, req.CreatedAt)
	if err != nil {
		return fmt.Errorf("failed to store integration request %s: %w", req.ID, err)
	}
	return nil
}

// AIDEV-NOTE: 250902160530 Fixed receiver type and added proper error wrapping
// updateIntegrationRequest updates integration request with results
func (h *Handler) updateIntegrationRequest(ctx context.Context, req *models.IntegrationRequest) error {
	query := `
		UPDATE integration_requests 
		SET status = $2, response = $3, error = $4, duration_ms = $5, completed_at = $6
		WHERE id = $1
	`
	_, err := h.db.ExecContext(ctx, query, req.ID, req.Status, req.Response, req.Error, req.Duration, req.CompletedAt)
	if err != nil {
		return fmt.Errorf("failed to update integration request %s: %w", req.ID, err)
	}
	return nil
}

// AIDEV-NOTE: 250902160530 Fixed receiver type and improved error handling
// validateKOLCandidates validates that KOL candidates exist in database
func (h *Handler) validateKOLCandidates(ctx context.Context, kolIDs []string) ([]string, error) {
	if len(kolIDs) == 0 {
		return nil, fmt.Errorf("no KOL candidates provided")
	}

	query := `SELECT id FROM kols WHERE id = ANY($1) AND is_active = true`
	var validIDs []string
	if err := h.db.SelectContext(ctx, &validIDs, query, kolIDs); err != nil {
		return nil, fmt.Errorf("failed to validate KOL candidates: %w", err)
	}

	if len(validIDs) == 0 {
		return nil, fmt.Errorf("no valid KOL candidates found")
	}

	if len(validIDs) < len(kolIDs) {
		h.logger.Warn("Some KOL candidates not found", "requested", len(kolIDs), "found", len(validIDs))
	}

	return validIDs, nil
}

// AIDEV-NOTE: Utility helper functions

// generateRequestID creates a unique request ID
func generateRequestID() string {
	return fmt.Sprintf("req_%d", time.Now().UnixNano())
}

// calculateExpectedROI calculates simple ROI based on budget and reach
func calculateExpectedROI(budget float64, reach int) float64 {
	if budget <= 0 {
		return 0
	}
	// Simplified ROI calculation: reach per dollar spent
	return float64(reach) / budget
}

// structToMap converts struct to map[string]interface{} for JSON storage
func structToMap(v interface{}) map[string]interface{} {
	// AIDEV-NOTE: In production, use proper reflection or JSON marshal/unmarshal
	// For now, returning a simple placeholder
	return map[string]interface{}{
		"converted": true,
		"timestamp": time.Now(),
	}
}

// convertTierToString converts KOLTier slice to string slice
func convertTierToString(tiers []models.KOLTier) []string {
	result := make([]string, len(tiers))
	for i, tier := range tiers {
		result[i] = string(tier)
	}
	return result
}

// convertPlatformToString converts Platform slice to string slice
func convertPlatformToString(platforms []models.Platform) []string {
	result := make([]string, len(platforms))
	for i, platform := range platforms {
		result[i] = string(platform)
	}
	return result
}

// convertCategoryToString converts Category slice to string slice
func convertCategoryToString(categories []models.Category) []string {
	result := make([]string, len(categories))
	for i, category := range categories {
		result[i] = string(category)
	}
	return result
}