// AIDEV-NOTE: 250102120320 Integration models for POC2/POC4 communication
// Defines request/response structures for KOL matching and budget optimization
package models

import (
	"fmt"
	"time"

	"github.com/google/uuid"
)

// AIDEV-NOTE: Integration request tracking for audit trails
type IntegrationRequest struct {
	ID          string    `json:"id" db:"id"`
	RequestType string    `json:"request_type" db:"request_type"`
	UserID      string    `json:"user_id,omitempty" db:"user_id"`
	Payload     JSONBMap  `json:"payload" db:"payload"`
	Status      string    `json:"status" db:"status"`
	Response    *JSONBMap `json:"response,omitempty" db:"response"`
	Error       *string   `json:"error,omitempty" db:"error"`
	Duration    *int      `json:"duration_ms,omitempty" db:"duration_ms"`
	CreatedAt   time.Time `json:"created_at" db:"created_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty" db:"completed_at"`
}

// BeforeCreate sets up integration request before creation
func (ir *IntegrationRequest) BeforeCreate() {
	if ir.ID == "" {
		ir.ID = uuid.New().String()
	}
	ir.CreatedAt = time.Now().UTC()
	if ir.Status == "" {
		ir.Status = "pending"
	}
}

// Complete marks the integration request as completed
func (ir *IntegrationRequest) Complete(response interface{}, duration time.Duration) {
	ir.Status = "completed"
	now := time.Now().UTC()
	ir.CompletedAt = &now
	durationMs := int(duration.Milliseconds())
	ir.Duration = &durationMs
	
	if response != nil {
		responseMap := JSONBMap(response.(map[string]interface{}))
		ir.Response = &responseMap
	}
}

// Fail marks the integration request as failed
func (ir *IntegrationRequest) Fail(err error, duration time.Duration) {
	ir.Status = "failed"
	now := time.Now().UTC()
	ir.CompletedAt = &now
	durationMs := int(duration.Milliseconds())
	ir.Duration = &durationMs
	errMsg := err.Error()
	ir.Error = &errMsg
}

// AIDEV-NOTE: KOL matching request validation
type KOLMatchingRequest struct {
	CampaignBrief  string                 `json:"campaign_brief" validate:"required,min=10,max=5000"`
	Budget         float64                `json:"budget" validate:"required,min=100"`
	TargetTier     []KOLTier             `json:"target_tier,omitempty"`
	Platforms      []Platform            `json:"platforms,omitempty"`
	Categories     []Category            `json:"categories,omitempty"`
	MinFollowers   int                   `json:"min_followers,omitempty" validate:"min=0"`
	MaxFollowers   int                   `json:"max_followers,omitempty" validate:"min=0"`
	Demographics   map[string]interface{} `json:"demographics,omitempty"`
	MaxResults     int                   `json:"max_results,omitempty" validate:"min=1,max=100"`
	RequestID      string                `json:"request_id,omitempty"`
}

// Validate validates the KOL matching request
func (kmr *KOLMatchingRequest) Validate() error {
	if kmr.CampaignBrief == "" {
		return ErrValidationFailed{Field: "campaign_brief", Message: "campaign brief is required"}
	}
	
	if len(kmr.CampaignBrief) < 10 {
		return ErrValidationFailed{Field: "campaign_brief", Message: "campaign brief must be at least 10 characters"}
	}
	
	if len(kmr.CampaignBrief) > 5000 {
		return ErrValidationFailed{Field: "campaign_brief", Message: "campaign brief must not exceed 5000 characters"}
	}
	
	if kmr.Budget < 100 {
		return ErrValidationFailed{Field: "budget", Message: "budget must be at least $100"}
	}
	
	if kmr.MaxResults <= 0 {
		kmr.MaxResults = 20 // Default
	}
	
	if kmr.MaxResults > 100 {
		return ErrValidationFailed{Field: "max_results", Message: "max_results cannot exceed 100"}
	}
	
	if kmr.MaxFollowers > 0 && kmr.MinFollowers > 0 && kmr.MinFollowers >= kmr.MaxFollowers {
		return ErrValidationFailed{Field: "follower_range", Message: "min_followers must be less than max_followers"}
	}
	
	return nil
}

// AIDEV-NOTE: Budget optimization request validation
type BudgetOptimizationRequest struct {
	TotalBudget      float64                `json:"total_budget" validate:"required,min=100"`
	CampaignGoals    []string              `json:"campaign_goals" validate:"required,min=1"`
	KOLCandidates    []string              `json:"kol_candidates" validate:"required,min=1"`
	TargetReach      int                   `json:"target_reach,omitempty" validate:"min=0"`
	TargetEngagement float64               `json:"target_engagement,omitempty" validate:"min=0,max=100"`
	Constraints      BudgetConstraints     `json:"constraints,omitempty"`
	RequestID        string                `json:"request_id,omitempty"`
}

type BudgetConstraints struct {
	MaxKOLs        int                    `json:"max_kols,omitempty" validate:"min=1,max=50"`
	MinKOLs        int                    `json:"min_kols,omitempty" validate:"min=1"`
	TierLimits     map[string]int         `json:"tier_limits,omitempty"`
	PlatformLimits map[string]float64     `json:"platform_limits,omitempty"`
}

// Validate validates the budget optimization request
func (bor *BudgetOptimizationRequest) Validate() error {
	if bor.TotalBudget < 100 {
		return ErrValidationFailed{Field: "total_budget", Message: "total budget must be at least $100"}
	}
	
	if len(bor.CampaignGoals) == 0 {
		return ErrValidationFailed{Field: "campaign_goals", Message: "at least one campaign goal is required"}
	}
	
	if len(bor.KOLCandidates) == 0 {
		return ErrValidationFailed{Field: "kol_candidates", Message: "at least one KOL candidate is required"}
	}
	
	if len(bor.KOLCandidates) > 1000 {
		return ErrValidationFailed{Field: "kol_candidates", Message: "cannot exceed 1000 KOL candidates"}
	}
	
	if bor.TargetEngagement < 0 || bor.TargetEngagement > 100 {
		return ErrValidationFailed{Field: "target_engagement", Message: "target engagement must be between 0 and 100"}
	}
	
	// Validate constraints
	if bor.Constraints.MinKOLs > 0 && bor.Constraints.MaxKOLs > 0 {
		if bor.Constraints.MinKOLs >= bor.Constraints.MaxKOLs {
			return ErrValidationFailed{Field: "constraints", Message: "min_kols must be less than max_kols"}
		}
	}
	
	return nil
}

// AIDEV-NOTE: Integration response structures
type IntegrationResponse struct {
	Success   bool        `json:"success"`
	Data      interface{} `json:"data,omitempty"`
	Error     string      `json:"error,omitempty"`
	Message   string      `json:"message,omitempty"`
	RequestID string      `json:"request_id,omitempty"`
	Meta      ResponseMeta `json:"meta,omitempty"`
}

type ResponseMeta struct {
	ProcessingTime time.Duration `json:"processing_time_ms"`
	Timestamp      time.Time     `json:"timestamp"`
	Version        string        `json:"version,omitempty"`
	CacheHit       bool          `json:"cache_hit,omitempty"`
}

// AIDEV-NOTE: Enhanced KOL matching response structures
type KOLMatchingResponse struct {
	Matches []EnhancedKOLMatch `json:"matches"`
	Meta    MatchingMeta       `json:"meta"`
}

type EnhancedKOLMatch struct {
	KOL            KOLSummary      `json:"kol"`
	Score          float64         `json:"score"`
	Reasoning      string          `json:"reasoning"`
	EstimatedCost  float64         `json:"estimated_cost"`
	Metrics        MatchMetrics    `json:"metrics"`
	ContentSample  []ContentPreview `json:"content_sample,omitempty"`
	Compatibility  CompatibilityScore `json:"compatibility"`
}

type KOLSummary struct {
	ID             string    `json:"id"`
	Username       string    `json:"username"`
	DisplayName    string    `json:"display_name"`
	Platform       Platform  `json:"platform"`
	Tier           KOLTier   `json:"tier"`
	Category       Category  `json:"category"`
	IsVerified     bool      `json:"is_verified"`
	IsBrandSafe    bool      `json:"is_brand_safe"`
	ProfileURL     string    `json:"profile_url"`
	AvatarURL      *string   `json:"avatar_url,omitempty"`
}

type MatchMetrics struct {
	FollowerCount     int     `json:"follower_count"`
	EngagementRate    float64 `json:"engagement_rate"`
	AvgLikes          float64 `json:"avg_likes"`
	AvgComments       float64 `json:"avg_comments"`
	PostsLast30Days   int     `json:"posts_last_30_days"`
	AudienceReach     int     `json:"audience_reach"`
	EstimatedViews    int     `json:"estimated_views"`
}

type ContentPreview struct {
	ID              string    `json:"id"`
	ContentType     string    `json:"content_type"`
	Caption         string    `json:"caption,omitempty"`
	LikesCount      int       `json:"likes_count"`
	CommentsCount   int       `json:"comments_count"`
	PostedAt        time.Time `json:"posted_at"`
	PerformanceScore float64  `json:"performance_score"`
}

type CompatibilityScore struct {
	Overall         float64 `json:"overall"`
	BrandAlignment  float64 `json:"brand_alignment"`
	AudienceMatch   float64 `json:"audience_match"`
	ContentQuality  float64 `json:"content_quality"`
	Engagement      float64 `json:"engagement"`
	ReliabilityScore float64 `json:"reliability_score"`
}

type MatchingMeta struct {
	TotalCandidates   int       `json:"total_candidates"`
	FilteredCount     int       `json:"filtered_count"`
	MatchesFound      int       `json:"matches_found"`
	QueryTime         float64   `json:"query_time_ms"`
	AlgorithmVersion  string    `json:"algorithm_version"`
	Confidence        float64   `json:"confidence"`
	SearchCriteria    map[string]interface{} `json:"search_criteria"`
}

// AIDEV-NOTE: Enhanced budget optimization response structures
type BudgetOptimizationResponse struct {
	Allocation []EnhancedBudgetAllocation `json:"allocation"`
	Summary    OptimizationSummary        `json:"summary"`
	Analysis   OptimizationAnalysis       `json:"analysis"`
	Meta       OptimizationMeta           `json:"meta"`
}

type EnhancedBudgetAllocation struct {
	KOL                 KOLSummary   `json:"kol"`
	AllocatedBudget     float64      `json:"allocated_budget"`
	Priority            int          `json:"priority"`
	ExpectedMetrics     ExpectedMetrics `json:"expected_metrics"`
	ROIProjection       ROIProjection `json:"roi_projection"`
	RiskAssessment      RiskAssessment `json:"risk_assessment"`
	Reasoning           string       `json:"reasoning"`
	AlternativeOptions  []AlternativeOption `json:"alternative_options,omitempty"`
}

type ExpectedMetrics struct {
	EstimatedReach      int     `json:"estimated_reach"`
	EstimatedEngagement float64 `json:"estimated_engagement"`
	EstimatedImpressions int    `json:"estimated_impressions"`
	EstimatedClicks     int     `json:"estimated_clicks"`
	EstimatedConversions int    `json:"estimated_conversions"`
}

type ROIProjection struct {
	ExpectedROI         float64 `json:"expected_roi"`
	BreakevenPoint      float64 `json:"breakeven_point"`
	ProjectedRevenue    float64 `json:"projected_revenue"`
	CostPerAcquisition  float64 `json:"cost_per_acquisition"`
	LifetimeValue       float64 `json:"lifetime_value"`
}

type RiskAssessment struct {
	OverallRisk         string  `json:"overall_risk"` // low, medium, high
	ReputationRisk      float64 `json:"reputation_risk"`
	PerformanceRisk     float64 `json:"performance_risk"`
	ComplianceRisk      float64 `json:"compliance_risk"`
	FinancialRisk       float64 `json:"financial_risk"`
	MitigationStrategy  string  `json:"mitigation_strategy,omitempty"`
}

type AlternativeOption struct {
	KOLID               string  `json:"kol_id"`
	Username            string  `json:"username"`
	AlternativeBudget   float64 `json:"alternative_budget"`
	TradeoffReason      string  `json:"tradeoff_reason"`
	ImpactOnOverallROI  float64 `json:"impact_on_overall_roi"`
}

type OptimizationSummary struct {
	TotalBudget             float64 `json:"total_budget"`
	TotalAllocated          float64 `json:"total_allocated"`
	RemainingBudget         float64 `json:"remaining_budget"`
	OptimalKOLsSelected     int     `json:"optimal_kols_selected"`
	BudgetUtilization       float64 `json:"budget_utilization"`
	ExpectedOverallReach    int     `json:"expected_overall_reach"`
	ExpectedOverallEngagement float64 `json:"expected_overall_engagement"`
	OverallEfficiencyScore  float64 `json:"overall_efficiency_score"`
	ProjectedOverallROI     float64 `json:"projected_overall_roi"`
}

type OptimizationAnalysis struct {
	StrategyRecommendation  string                 `json:"strategy_recommendation"`
	KeyInsights            []string               `json:"key_insights"`
	PerformanceForecasting PerformanceForecasting `json:"performance_forecasting"`
	CompetitiveAnalysis    CompetitiveAnalysis    `json:"competitive_analysis"`
	RecommendedAdjustments []RecommendedAdjustment `json:"recommended_adjustments"`
}

type PerformanceForecasting struct {
	WeeklyProjections  []WeeklyProjection `json:"weekly_projections"`
	SeasonalFactors    map[string]float64 `json:"seasonal_factors"`
	ConfidenceInterval map[string]float64 `json:"confidence_interval"`
}

type WeeklyProjection struct {
	Week                int     `json:"week"`
	ProjectedReach      int     `json:"projected_reach"`
	ProjectedEngagement float64 `json:"projected_engagement"`
	ProjectedConversions int    `json:"projected_conversions"`
}

type CompetitiveAnalysis struct {
	MarketPosition      string  `json:"market_position"`
	CompetitorComparison map[string]float64 `json:"competitor_comparison"`
	MarketSharePotential float64 `json:"market_share_potential"`
}

type RecommendedAdjustment struct {
	Type        string  `json:"type"` // budget_reallocation, kol_substitution, timeline_adjustment
	Description string  `json:"description"`
	Impact      float64 `json:"impact"`
	Priority    string  `json:"priority"` // high, medium, low
}

type OptimizationMeta struct {
	OptimizationTime    float64 `json:"optimization_time_ms"`
	AlgorithmVersion    string  `json:"algorithm_version"`
	ModelAccuracy       float64 `json:"model_accuracy"`
	DataFreshness       time.Time `json:"data_freshness"`
	IterationsRun       int     `json:"iterations_run"`
	ConvergenceAchieved bool    `json:"convergence_achieved"`
}

// AIDEV-NOTE: Validation error types
type ErrValidationFailed struct {
	Field   string
	Message string
}

func (e ErrValidationFailed) Error() string {
	return fmt.Sprintf("validation failed for field '%s': %s", e.Field, e.Message)
}

// AIDEV-NOTE: Integration status constants
const (
	IntegrationStatusPending   = "pending"
	IntegrationStatusRunning   = "running"
	IntegrationStatusCompleted = "completed"
	IntegrationStatusFailed    = "failed"
	IntegrationStatusTimeout   = "timeout"
)

// AIDEV-NOTE: Request type constants
const (
	RequestTypeMatchKOLs       = "match_kols"
	RequestTypeOptimizeBudget  = "optimize_budget"
	RequestTypeHealthCheck     = "health_check"
)

// AIDEV-NOTE: Campaign goals enum
var ValidCampaignGoals = map[string]bool{
	"brand_awareness":    true,
	"lead_generation":    true,
	"sales":             true,
	"engagement":        true,
	"reach":             true,
	"conversions":       true,
	"app_downloads":     true,
	"website_traffic":   true,
	"social_following":  true,
	"content_creation":  true,
}

// AIDEV-NOTE: Risk levels enum
var RiskLevels = map[string]int{
	"low":    1,
	"medium": 2,
	"high":   3,
}

// IsValidCampaignGoal checks if the campaign goal is valid
func IsValidCampaignGoal(goal string) bool {
	return ValidCampaignGoals[goal]
}

// GetRiskLevel returns numeric risk level
func GetRiskLevel(risk string) int {
	if level, exists := RiskLevels[risk]; exists {
		return level
	}
	return 2 // default to medium
}