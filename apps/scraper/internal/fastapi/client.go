// AIDEV-NOTE: 250102120315 FastAPI GraphQL client for POC2/POC4 integration
// High-performance client with connection pooling and proper error handling for KOL matching and budget optimization
package fastapi

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/machinebox/graphql"
	"github.com/pkg/errors"
	"kol-scraper/internal/circuit"
	"kol-scraper/pkg/config"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: 250902160600 Enhanced Client with circuit breaker and improved connection pooling
type Client struct {
	httpClient    *http.Client
	graphqlClient *graphql.Client
	baseURL       string
	logger        *logger.Logger
	circuitBreaker *circuit.CircuitBreaker
	// Performance monitoring
	totalRequests   uint64
	successRequests uint64
	failedRequests  uint64
}

// AIDEV-NOTE: Request/response structures for POC2 KOL matching
type MatchKOLsRequest struct {
	CampaignBrief  string            `json:"campaign_brief"`
	Budget         float64           `json:"budget"`
	TargetTier     []string          `json:"target_tier,omitempty"`
	Platforms      []string          `json:"platforms,omitempty"`
	Categories     []string          `json:"categories,omitempty"`
	MinFollowers   int               `json:"min_followers,omitempty"`
	MaxFollowers   int               `json:"max_followers,omitempty"`
	Demographics   map[string]interface{} `json:"demographics,omitempty"`
	MaxResults     int               `json:"max_results,omitempty"`
}

type MatchKOLsResponse struct {
	Matches []KOLMatch `json:"matches"`
	Meta    MatchMeta  `json:"meta"`
}

type KOLMatch struct {
	KOLID          string  `json:"kol_id"`
	Username       string  `json:"username"`
	Platform       string  `json:"platform"`
	DisplayName    string  `json:"display_name"`
	FollowerCount  int     `json:"follower_count"`
	EngagementRate float64 `json:"engagement_rate"`
	Score          float64 `json:"score"`
	Tier           string  `json:"tier"`
	Category       string  `json:"category"`
	EstimatedCost  float64 `json:"estimated_cost"`
	Reasoning      string  `json:"reasoning"`
}

type MatchMeta struct {
	TotalFound     int     `json:"total_found"`
	QueryTime      float64 `json:"query_time_ms"`
	AlgorithmUsed  string  `json:"algorithm_used"`
	Confidence     float64 `json:"confidence"`
}

// AIDEV-NOTE: Request/response structures for POC4 budget optimization
type OptimizeBudgetRequest struct {
	TotalBudget      float64   `json:"total_budget"`
	CampaignGoals    []string  `json:"campaign_goals"`
	KOLCandidates    []string  `json:"kol_candidates"` // KOL IDs
	TargetReach      int       `json:"target_reach,omitempty"`
	TargetEngagement float64   `json:"target_engagement,omitempty"`
	Constraints      BudgetConstraints `json:"constraints,omitempty"`
}

type BudgetConstraints struct {
	MaxKOLs        int               `json:"max_kols,omitempty"`
	MinKOLs        int               `json:"min_kols,omitempty"`
	TierLimits     map[string]int    `json:"tier_limits,omitempty"`
	PlatformLimits map[string]float64 `json:"platform_limits,omitempty"`
}

type OptimizeBudgetResponse struct {
	Allocation []BudgetAllocation `json:"allocation"`
	Summary    BudgetSummary      `json:"summary"`
	Meta       OptimizationMeta   `json:"meta"`
}

type BudgetAllocation struct {
	KOLID           string  `json:"kol_id"`
	Username        string  `json:"username"`
	Platform        string  `json:"platform"`
	AllocatedBudget float64 `json:"allocated_budget"`
	EstimatedReach  int     `json:"estimated_reach"`
	EstimatedEngagement float64 `json:"estimated_engagement"`
	Priority        int     `json:"priority"`
	Reasoning       string  `json:"reasoning"`
}

type BudgetSummary struct {
	TotalAllocated      float64 `json:"total_allocated"`
	RemainingBudget     float64 `json:"remaining_budget"`
	EstimatedTotalReach int     `json:"estimated_total_reach"`
	EstimatedAvgEngagement float64 `json:"estimated_avg_engagement"`
	OptimalKOLsSelected int     `json:"optimal_kols_selected"`
	EfficiencyScore     float64 `json:"efficiency_score"`
}

type OptimizationMeta struct {
	OptimizationTime float64 `json:"optimization_time_ms"`
	AlgorithmUsed    string  `json:"algorithm_used"`
	Iterations       int     `json:"iterations"`
	Confidence       float64 `json:"confidence"`
}

// AIDEV-NOTE: GraphQL query definitions
const (
	matchKOLsQuery = `
		mutation MatchKOLs($input: MatchKOLsInput!) {
			matchKOLs(input: $input) {
				matches {
					kolId
					username
					platform
					displayName
					followerCount
					engagementRate
					score
					tier
					category
					estimatedCost
					reasoning
				}
				meta {
					totalFound
					queryTime
					algorithmUsed
					confidence
				}
			}
		}
	`

	optimizeBudgetQuery = `
		mutation OptimizeBudget($input: OptimizeBudgetInput!) {
			optimizeBudget(input: $input) {
				allocation {
					kolId
					username
					platform
					allocatedBudget
					estimatedReach
					estimatedEngagement
					priority
					reasoning
				}
				summary {
					totalAllocated
					remainingBudget
					estimatedTotalReach
					estimatedAvgEngagement
					optimalKOLsSelected
					efficiencyScore
				}
				meta {
					optimizationTime
					algorithmUsed
					iterations
					confidence
				}
			}
		}
	`
)

// AIDEV-NOTE: 250903161030 NewClient creates a new FastAPI client with enhanced circuit breaker and resilience features
func NewClient(cfg *config.Config, log *logger.Logger) (*Client, error) {
	// AIDEV-NOTE: Configure HTTP client with optimized connection pooling and timeouts
	dialer := &net.Dialer{
		Timeout:   5 * time.Second,
		KeepAlive: 30 * time.Second,
	}

	transport := &http.Transport{
		DialContext:            dialer.DialContext,
		MaxIdleConns:           200,  // Increased for better performance
		MaxIdleConnsPerHost:    50,   // Increased for better concurrency
		MaxConnsPerHost:        100,  // Limit total connections per host
		IdleConnTimeout:        90 * time.Second,
		TLSHandshakeTimeout:    10 * time.Second,
		ExpectContinueTimeout:  1 * time.Second,
		ResponseHeaderTimeout:  30 * time.Second,
		DisableCompression:     false,
		ForceAttemptHTTP2:      true, // Enable HTTP/2
		TLSClientConfig: &tls.Config{
			InsecureSkipVerify: false,
			MinVersion:         tls.VersionTLS12,
		},
	}

	httpClient := &http.Client{
		Timeout:   time.Duration(cfg.APITimeout) * time.Second,
		Transport: transport,
	}

	// AIDEV-NOTE: 250102161030 Create enhanced circuit breaker with custom configuration
	cbConfig := circuit.DefaultConfig()
	// Customize based on config if available
	if cfg.CircuitBreakerEnabled {
		if cfg.MaxFailures > 0 {
			cbConfig.MaxFailures = cfg.MaxFailures
		}
		if cfg.CircuitBreakerTimeout > 0 {
			cbConfig.RecoveryTimeout = time.Duration(cfg.CircuitBreakerTimeout) * time.Second
		}
	}
	
	cb := circuit.NewCircuitBreaker(cbConfig)
	
	// AIDEV-NOTE: 250102161030 Set up fallback function for graceful degradation
	cb.SetFallback(func(err error) (interface{}, error) {
		log.Warn("Circuit breaker open, using fallback response", "error", err.Error())
		return &MatchKOLsResponse{
			Matches: []KOLMatch{},
			Meta: MatchMeta{
				TotalFound:    0,
				QueryTime:     0,
				AlgorithmUsed: "fallback",
				Confidence:    0.0,
			},
		}, nil
	})
	
	// AIDEV-NOTE: Set up state change callback for monitoring
	cb.SetOnStateChange(func(from, to circuit.State) {
		log.IntegrationLog("circuit_breaker", "state_changed", logger.Fields{
			"from_state": from.String(),
			"to_state":   to.String(),
		})
	})

	// AIDEV-NOTE: Create GraphQL client with custom HTTP client
	graphqlClient := graphql.NewClient(cfg.FastAPIURL + "/graphql")
	graphqlClient.WithHTTPClient(httpClient)
	
	client := &Client{
		httpClient:     httpClient,
		graphqlClient:  graphqlClient,
		baseURL:        cfg.FastAPIURL,
		logger:         log,
		circuitBreaker: cb,
	}

	// AIDEV-NOTE: Test connectivity on initialization with circuit breaker
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := client.HealthCheck(ctx); err != nil {
		log.ErrorLog(err, "fastapi_client_init", logger.Fields{
			"base_url": cfg.FastAPIURL,
			"circuit_config": map[string]interface{}{
				"max_failures":      cbConfig.MaxFailures,
				"failure_threshold": cbConfig.FailureThreshold,
				"recovery_timeout":  cbConfig.RecoveryTimeout.String(),
			},
		})
		// AIDEV-NOTE: Don't fail initialization if health check fails - let circuit breaker handle it
		log.Warn("FastAPI health check failed during initialization, proceeding with circuit breaker protection")
	}

	log.Info("FastAPI client initialized successfully with enhanced circuit breaker", "base_url", cfg.FastAPIURL)
	return client, nil
}

// AIDEV-NOTE: HealthCheck verifies FastAPI backend connectivity
func (c *Client) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/health", nil)
	if err != nil {
		return errors.Wrap(err, "failed to create health check request")
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "KOL-Scraper/1.0")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return errors.Wrap(err, "health check request failed")
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("health check failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// AIDEV-NOTE: 250903161030 MatchKOLs calls POC2 KOL matching algorithm via GraphQL with enhanced circuit breaker
func (c *Client) MatchKOLs(ctx context.Context, req *MatchKOLsRequest) (*MatchKOLsResponse, error) {
	c.totalRequests++
	metrics := c.circuitBreaker.GetMetrics()
	c.logger.IntegrationLog("fastapi", "match_kols_start", logger.Fields{
		"budget":                req.Budget,
		"platforms":             req.Platforms,
		"max_results":           req.MaxResults,
		"circuit_state":         metrics.State.String(),
		"circuit_failures":      metrics.FailureCount,
		"circuit_success_rate":  metrics.SuccessRate,
		"total_requests":        c.totalRequests,
		"current_backoff":       metrics.CurrentBackoff.String(),
		"consecutive_failures":  metrics.ConsecutiveFailures,
	})

	startTime := time.Now()

	// AIDEV-NOTE: Create GraphQL request with variables
	request := graphql.NewRequest(matchKOLsQuery)
	request.Var("input", map[string]interface{}{
		"campaignBrief":  req.CampaignBrief,
		"budget":         req.Budget,
		"targetTier":     req.TargetTier,
		"platforms":      req.Platforms,
		"categories":     req.Categories,
		"minFollowers":   req.MinFollowers,
		"maxFollowers":   req.MaxFollowers,
		"demographics":   req.Demographics,
		"maxResults":     req.MaxResults,
	})

	// AIDEV-NOTE: Set request headers with tracing
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("User-Agent", "KOL-Scraper/1.0")
	request.Header.Set("X-Request-ID", fmt.Sprintf("match-kols-%d", time.Now().UnixNano()))
	request.Header.Set("X-Circuit-State", metrics.State.String())

	// AIDEV-NOTE: Execute GraphQL request with enhanced circuit breaker
	var response struct {
		MatchKOLs MatchKOLsResponse `json:"matchKOLs"`
	}

	result, err := c.circuitBreaker.Execute(ctx, func() (interface{}, error) {
		return &response, c.graphqlClient.Run(ctx, request, &response)
	})

	if err != nil {
		c.failedRequests++
		updatedMetrics := c.circuitBreaker.GetMetrics()
		c.logger.ErrorLog(err, "fastapi_match_kols_failed", logger.Fields{
			"duration_ms":       time.Since(startTime).Milliseconds(),
			"circuit_state":     updatedMetrics.State,
			"circuit_failures":  updatedMetrics.TotalFailures,
			"failed_requests":   c.failedRequests,
			"consecutive_fails": updatedMetrics.ConsecutiveFailures,
			"next_retry_backoff": updatedMetrics.CurrentBackoff.String(),
		})
		
		// AIDEV-NOTE: Check if this was a fallback response
		if result != nil {
			if fallbackResp, ok := result.(*MatchKOLsResponse); ok {
				c.logger.Info("Using circuit breaker fallback response for MatchKOLs")
				return fallbackResp, nil
			}
		}
		
		return nil, errors.Wrap(err, "GraphQL MatchKOLs request failed")
	}

	response = *result.(*struct {
		MatchKOLs MatchKOLsResponse `json:"matchKOLs"`
	})

	c.successRequests++
	duration := time.Since(startTime)
	finalMetrics := c.circuitBreaker.GetMetrics()
	c.logger.IntegrationLog("fastapi", "match_kols_completed", logger.Fields{
		"matches_found":        len(response.MatchKOLs.Matches),
		"duration_ms":          duration.Milliseconds(),
		"confidence":           response.MatchKOLs.Meta.Confidence,
		"circuit_state":        finalMetrics.State,
		"circuit_success_rate": finalMetrics.SuccessRate,
		"success_requests":     c.successRequests,
		"client_success_rate":  float64(c.successRequests) / float64(c.totalRequests),
	})

	return &response.MatchKOLs, nil
}

// AIDEV-NOTE: 250903161030 OptimizeBudget calls POC4 budget optimization algorithm via GraphQL with enhanced circuit breaker
func (c *Client) OptimizeBudget(ctx context.Context, req *OptimizeBudgetRequest) (*OptimizeBudgetResponse, error) {
	c.totalRequests++
	metrics := c.circuitBreaker.GetMetrics()
	c.logger.IntegrationLog("fastapi", "optimize_budget_start", logger.Fields{
		"total_budget":         req.TotalBudget,
		"campaign_goals":       req.CampaignGoals,
		"kol_candidates":       len(req.KOLCandidates),
		"circuit_state":        metrics.State,
		"circuit_failures":     metrics.TotalFailures,
		"circuit_success_rate": metrics.SuccessRate,
		"total_requests":       c.totalRequests,
		"current_backoff":      metrics.CurrentBackoff.String(),
	})

	startTime := time.Now()

	// AIDEV-NOTE: Create GraphQL request with variables
	request := graphql.NewRequest(optimizeBudgetQuery)
	request.Var("input", map[string]interface{}{
		"totalBudget":      req.TotalBudget,
		"campaignGoals":    req.CampaignGoals,
		"kolCandidates":    req.KOLCandidates,
		"targetReach":      req.TargetReach,
		"targetEngagement": req.TargetEngagement,
		"constraints":      req.Constraints,
	})

	// AIDEV-NOTE: Set request headers with tracing
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("User-Agent", "KOL-Scraper/1.0")
	request.Header.Set("X-Request-ID", fmt.Sprintf("optimize-budget-%d", time.Now().UnixNano()))
	request.Header.Set("X-Circuit-State", metrics.State.String())

	// AIDEV-NOTE: Execute GraphQL request with enhanced circuit breaker and fallback
	var response struct {
		OptimizeBudget OptimizeBudgetResponse `json:"optimizeBudget"`
	}

	result, err := c.circuitBreaker.Execute(ctx, func() (interface{}, error) {
		return &response, c.graphqlClient.Run(ctx, request, &response)
	})

	if err != nil {
		c.failedRequests++
		updatedMetrics := c.circuitBreaker.GetMetrics()
		c.logger.ErrorLog(err, "fastapi_optimize_budget_failed", logger.Fields{
			"duration_ms":       time.Since(startTime).Milliseconds(),
			"circuit_state":     updatedMetrics.State,
			"circuit_failures":  updatedMetrics.TotalFailures,
			"failed_requests":   c.failedRequests,
			"consecutive_fails": updatedMetrics.ConsecutiveFailures,
			"next_retry_backoff": updatedMetrics.CurrentBackoff.String(),
		})
		
		// AIDEV-NOTE: Provide fallback budget optimization when circuit is open
		if updatedMetrics.State == "open" {
			c.logger.Info("Circuit breaker open, providing conservative budget allocation fallback")
			fallbackResponse := &OptimizeBudgetResponse{
				Allocation: []BudgetAllocation{{
					KOLID:               "fallback-allocation",
					Username:            "conservative-approach",
					Platform:            "mixed",
					AllocatedBudget:     req.TotalBudget * 0.8, // Conservative 80% allocation
					EstimatedReach:      0,
					EstimatedEngagement: 0,
					Priority:            1,
					Reasoning:           "Fallback allocation due to service unavailability",
				}},
				Summary: BudgetSummary{
					TotalAllocated:         req.TotalBudget * 0.8,
					RemainingBudget:        req.TotalBudget * 0.2,
					EstimatedTotalReach:    0,
					EstimatedAvgEngagement: 0,
					OptimalKOLsSelected:    1,
					EfficiencyScore:        0.5, // Conservative score
				},
				Meta: OptimizationMeta{
					OptimizationTime: 0,
					AlgorithmUsed:    "fallback-conservative",
					Iterations:       0,
					Confidence:       0.5,
				},
			}
			return fallbackResponse, nil
		}
		
		return nil, errors.Wrap(err, "GraphQL OptimizeBudget request failed")
	}

	response = *result.(*struct {
		OptimizeBudget OptimizeBudgetResponse `json:"optimizeBudget"`
	})

	c.successRequests++
	duration := time.Since(startTime)
	finalMetrics := c.circuitBreaker.GetMetrics()
	c.logger.IntegrationLog("fastapi", "optimize_budget_completed", logger.Fields{
		"allocations_created":  len(response.OptimizeBudget.Allocation),
		"total_allocated":      response.OptimizeBudget.Summary.TotalAllocated,
		"duration_ms":          duration.Milliseconds(),
		"efficiency_score":     response.OptimizeBudget.Summary.EfficiencyScore,
		"circuit_state":        finalMetrics.State,
		"circuit_success_rate": finalMetrics.SuccessRate,
		"success_requests":     c.successRequests,
		"client_success_rate":  float64(c.successRequests) / float64(c.totalRequests),
	})

	return &response.OptimizeBudget, nil
}

// AIDEV-NOTE: 250903161030 Enhanced client management methods

// GetCircuitBreakerMetrics returns detailed circuit breaker metrics for monitoring
func (c *Client) GetCircuitBreakerMetrics() *circuit.Metrics {
	return c.circuitBreaker.GetMetrics()
}

// GetClientStats returns client performance statistics
func (c *Client) GetClientStats() map[string]interface{} {
	metrics := c.circuitBreaker.GetMetrics()
	return map[string]interface{}{
		"total_requests":     c.totalRequests,
		"success_requests":   c.successRequests,
		"failed_requests":    c.failedRequests,
		"success_rate":       float64(c.successRequests) / float64(c.totalRequests),
		"circuit_breaker":    metrics,
		"base_url":          c.baseURL,
	}
}

// ResetCircuitBreaker resets the circuit breaker to initial state (useful for testing)
func (c *Client) ResetCircuitBreaker() {
	c.circuitBreaker.Reset()
	c.totalRequests = 0
	c.successRequests = 0
	c.failedRequests = 0
	c.logger.Info("FastAPI client circuit breaker and stats reset")
}

// IsHealthy returns true if the circuit breaker is closed (service is healthy)
func (c *Client) IsHealthy() bool {
	return c.circuitBreaker.State() == circuit.StateClosed
}

// Close closes the HTTP client connections
func (c *Client) Close() error {
	if transport, ok := c.httpClient.Transport.(*http.Transport); ok {
		transport.CloseIdleConnections()
	}
	return nil
}

// AIDEV-NOTE: Error types for better error handling
type FastAPIError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

func (e *FastAPIError) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("FastAPI error %d: %s (%s)", e.Code, e.Message, e.Details)
	}
	return fmt.Sprintf("FastAPI error %d: %s", e.Code, e.Message)
}

// AIDEV-NOTE: Helper function to make HTTP requests with retry logic
func (c *Client) makeHTTPRequest(ctx context.Context, method, endpoint string, body interface{}) (*http.Response, error) {
	var reqBody io.Reader
	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, errors.Wrap(err, "failed to marshal request body")
		}
		reqBody = bytes.NewReader(jsonBody)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+endpoint, reqBody)
	if err != nil {
		return nil, errors.Wrap(err, "failed to create HTTP request")
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "KOL-Scraper/1.0")

	// AIDEV-NOTE: Execute request with retry logic
	maxRetries := 3
	for attempt := 0; attempt < maxRetries; attempt++ {
		resp, err := c.httpClient.Do(req)
		if err != nil {
			if attempt == maxRetries-1 {
				return nil, errors.Wrapf(err, "HTTP request failed after %d attempts", maxRetries)
			}
			
			// AIDEV-NOTE: Exponential backoff
			backoff := time.Duration(attempt+1) * time.Second
			c.logger.ErrorLog(err, "fastapi_request_retry", logger.Fields{
				"attempt":  attempt + 1,
				"backoff":  backoff.String(),
				"endpoint": endpoint,
			})
			
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoff):
				continue
			}
		}

		// AIDEV-NOTE: Check for HTTP errors
		if resp.StatusCode >= 400 {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			
			var apiErr FastAPIError
			if err := json.Unmarshal(body, &apiErr); err == nil {
				return nil, &apiErr
			}
			
			return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
		}

		return resp, nil
	}

	return nil, fmt.Errorf("unexpected retry loop exit")
}