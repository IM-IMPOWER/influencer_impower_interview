// AIDEV-NOTE: 250102120335 Tests for FastAPI GraphQL client
// Comprehensive tests with table-driven patterns for POC2/POC4 client functionality
package fastapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"kol-scraper/pkg/config"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: Test fixtures
var testConfig = &config.Config{
	FastAPIURL: "http://localhost:8000",
}

var testLogger = logger.New("debug", "test")

// AIDEV-NOTE: Mock GraphQL responses
var mockMatchKOLsGraphQLResponse = map[string]interface{}{
	"data": map[string]interface{}{
		"matchKOLs": map[string]interface{}{
			"matches": []interface{}{
				map[string]interface{}{
					"kolId":          "kol-123",
					"username":       "testuser",
					"platform":       "tiktok",
					"displayName":    "Test User",
					"followerCount":  50000,
					"engagementRate": 5.5,
					"score":          8.0,
					"tier":           "micro",
					"category":       "tech",
					"estimatedCost":  1500.0,
					"reasoning":      "Good match for tech content",
				},
			},
			"meta": map[string]interface{}{
				"totalFound":    10,
				"queryTime":     120.5,
				"algorithmUsed": "ml_matching_v2",
				"confidence":    0.85,
			},
		},
	},
}

var mockOptimizeBudgetGraphQLResponse = map[string]interface{}{
	"data": map[string]interface{}{
		"optimizeBudget": map[string]interface{}{
			"allocation": []interface{}{
				map[string]interface{}{
					"kolId":               "kol-123",
					"username":            "testuser",
					"platform":            "tiktok",
					"allocatedBudget":     3000.0,
					"estimatedReach":      150000,
					"estimatedEngagement": 5.2,
					"priority":            1,
					"reasoning":           "High ROI potential",
				},
			},
			"summary": map[string]interface{}{
				"totalAllocated":         3000.0,
				"remainingBudget":        2000.0,
				"estimatedTotalReach":    150000,
				"estimatedAvgEngagement": 5.2,
				"optimalKOLsSelected":    1,
				"efficiencyScore":        0.88,
			},
			"meta": map[string]interface{}{
				"optimizationTime": 250.3,
				"algorithmUsed":    "genetic_optimization_v3",
				"iterations":       12,
				"confidence":       0.89,
			},
		},
	},
}

// AIDEV-NOTE: Test server setup
func setupTestServer(responses map[string]interface{}) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"status": "healthy",
				"service": "fastapi-backend",
			})
		case "/graphql":
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(responses)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

// AIDEV-NOTE: Test Client creation and health check
func TestNewClient(t *testing.T) {
	tests := []struct {
		name        string
		setupServer func() *httptest.Server
		expectError bool
		errorMsg    string
	}{
		{
			name: "successful_client_creation",
			setupServer: func() *httptest.Server {
				return setupTestServer(nil)
			},
			expectError: false,
		},
		{
			name: "health_check_failure",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.WriteHeader(http.StatusServiceUnavailable)
				}))
			},
			expectError: true,
			errorMsg:    "failed to initialize FastAPI client",
		},
		{
			name: "connection_refused",
			setupServer: func() *httptest.Server {
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
				server.Close() // Close immediately to simulate connection refused
				return server
			},
			expectError: true,
			errorMsg:    "failed to initialize FastAPI client",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := tt.setupServer()
			if !tt.expectError {
				defer server.Close()
			}

			cfg := &config.Config{
				FastAPIURL: server.URL,
			}

			client, err := NewClient(cfg, testLogger)

			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
				assert.Nil(t, client)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, client)
				defer client.Close()
			}
		})
	}
}

// AIDEV-NOTE: Test MatchKOLs functionality
func TestClient_MatchKOLs(t *testing.T) {
	tests := []struct {
		name           string
		request        *MatchKOLsRequest
		serverResponse map[string]interface{}
		serverStatus   int
		expectError    bool
		errorMsg       string
		validateResponse func(*testing.T, *MatchKOLsResponse)
	}{
		{
			name: "successful_kol_matching",
			request: &MatchKOLsRequest{
				CampaignBrief:  "Test campaign",
				Budget:         5000.0,
				TargetTier:     []string{"micro", "mid"},
				Platforms:      []string{"tiktok"},
				Categories:     []string{"tech"},
				MinFollowers:   10000,
				MaxFollowers:   100000,
				MaxResults:     10,
			},
			serverResponse: mockMatchKOLsGraphQLResponse,
			serverStatus:   http.StatusOK,
			expectError:    false,
			validateResponse: func(t *testing.T, response *MatchKOLsResponse) {
				assert.Len(t, response.Matches, 1)
				
				match := response.Matches[0]
				assert.Equal(t, "kol-123", match.KOLID)
				assert.Equal(t, "testuser", match.Username)
				assert.Equal(t, "tiktok", match.Platform)
				assert.Equal(t, 50000, match.FollowerCount)
				assert.Equal(t, 5.5, match.EngagementRate)
				assert.Equal(t, 8.0, match.Score)
				
				assert.Equal(t, 10, response.Meta.TotalFound)
				assert.Equal(t, 120.5, response.Meta.QueryTime)
				assert.Equal(t, "ml_matching_v2", response.Meta.AlgorithmUsed)
				assert.Equal(t, 0.85, response.Meta.Confidence)
			},
		},
		{
			name: "graphql_error_response",
			request: &MatchKOLsRequest{
				CampaignBrief: "Test campaign",
				Budget:        5000.0,
				MaxResults:    10,
			},
			serverResponse: map[string]interface{}{
				"errors": []interface{}{
					map[string]interface{}{
						"message": "Invalid input parameters",
					},
				},
			},
			serverStatus: http.StatusOK, // GraphQL errors still return 200
			expectError:  true,
			errorMsg:     "GraphQL MatchKOLs request failed",
		},
		{
			name: "server_error",
			request: &MatchKOLsRequest{
				CampaignBrief: "Test campaign",
				Budget:        5000.0,
				MaxResults:    10,
			},
			serverResponse: nil,
			serverStatus:   http.StatusInternalServerError,
			expectError:    true,
			errorMsg:       "GraphQL MatchKOLs request failed",
		},
		{
			name: "context_timeout",
			request: &MatchKOLsRequest{
				CampaignBrief: "Test campaign",
				Budget:        5000.0,
				MaxResults:    10,
			},
			serverResponse: mockMatchKOLsGraphQLResponse,
			serverStatus:   http.StatusOK,
			expectError:    true,
			errorMsg:       "context deadline exceeded",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := setupTestServer(tt.serverResponse)
			defer server.Close()

			cfg := &config.Config{
				FastAPIURL: server.URL,
			}

			client, err := NewClient(cfg, testLogger)
			require.NoError(t, err)
			defer client.Close()

			// AIDEV-NOTE: Setup context with timeout for timeout test
			ctx := context.Background()
			if tt.name == "context_timeout" {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, 1*time.Microsecond)
				defer cancel()
				time.Sleep(2 * time.Microsecond) // Ensure timeout occurs
			}

			response, err := client.MatchKOLs(ctx, tt.request)

			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
				assert.Nil(t, response)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, response)
				if tt.validateResponse != nil {
					tt.validateResponse(t, response)
				}
			}
		})
	}
}

// AIDEV-NOTE: Test OptimizeBudget functionality  
func TestClient_OptimizeBudget(t *testing.T) {
	tests := []struct {
		name           string
		request        *OptimizeBudgetRequest
		serverResponse map[string]interface{}
		expectError    bool
		errorMsg       string
		validateResponse func(*testing.T, *OptimizeBudgetResponse)
	}{
		{
			name: "successful_budget_optimization",
			request: &OptimizeBudgetRequest{
				TotalBudget:      5000.0,
				CampaignGoals:    []string{"brand_awareness", "engagement"},
				KOLCandidates:    []string{"kol-123", "kol-456"},
				TargetReach:      200000,
				TargetEngagement: 5.0,
				Constraints: BudgetConstraints{
					MaxKOLs: 3,
					MinKOLs: 1,
				},
			},
			serverResponse: mockOptimizeBudgetGraphQLResponse,
			expectError:    false,
			validateResponse: func(t *testing.T, response *OptimizeBudgetResponse) {
				assert.Len(t, response.Allocation, 1)
				
				allocation := response.Allocation[0]
				assert.Equal(t, "kol-123", allocation.KOLID)
				assert.Equal(t, "testuser", allocation.Username)
				assert.Equal(t, "tiktok", allocation.Platform)
				assert.Equal(t, 3000.0, allocation.AllocatedBudget)
				assert.Equal(t, 150000, allocation.EstimatedReach)
				assert.Equal(t, 5.2, allocation.EstimatedEngagement)
				assert.Equal(t, 1, allocation.Priority)
				
				assert.Equal(t, 3000.0, response.Summary.TotalAllocated)
				assert.Equal(t, 2000.0, response.Summary.RemainingBudget)
				assert.Equal(t, 150000, response.Summary.EstimatedTotalReach)
				assert.Equal(t, 0.88, response.Summary.EfficiencyScore)
				
				assert.Equal(t, 250.3, response.Meta.OptimizationTime)
				assert.Equal(t, "genetic_optimization_v3", response.Meta.AlgorithmUsed)
				assert.Equal(t, 12, response.Meta.Iterations)
				assert.Equal(t, 0.89, response.Meta.Confidence)
			},
		},
		{
			name: "empty_kol_candidates",
			request: &OptimizeBudgetRequest{
				TotalBudget:   5000.0,
				CampaignGoals: []string{"brand_awareness"},
				KOLCandidates: []string{},
			},
			serverResponse: map[string]interface{}{
				"errors": []interface{}{
					map[string]interface{}{
						"message": "No KOL candidates provided",
					},
				},
			},
			expectError: true,
			errorMsg:    "GraphQL OptimizeBudget request failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := setupTestServer(tt.serverResponse)
			defer server.Close()

			cfg := &config.Config{
				FastAPIURL: server.URL,
			}

			client, err := NewClient(cfg, testLogger)
			require.NoError(t, err)
			defer client.Close()

			ctx := context.Background()
			response, err := client.OptimizeBudget(ctx, tt.request)

			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
				assert.Nil(t, response)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, response)
				if tt.validateResponse != nil {
					tt.validateResponse(t, response)
				}
			}
		})
	}
}

// AIDEV-NOTE: Test HealthCheck functionality
func TestClient_HealthCheck(t *testing.T) {
	tests := []struct {
		name         string
		serverStatus int
		expectError  bool
		errorMsg     string
	}{
		{
			name:         "healthy_server",
			serverStatus: http.StatusOK,
			expectError:  false,
		},
		{
			name:         "unhealthy_server",
			serverStatus: http.StatusServiceUnavailable,
			expectError:  true,
			errorMsg:     "health check failed with status 503",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/health" {
					w.WriteHeader(tt.serverStatus)
					if tt.serverStatus == http.StatusOK {
						json.NewEncoder(w).Encode(map[string]interface{}{
							"status": "healthy",
						})
					} else {
						json.NewEncoder(w).Encode(map[string]interface{}{
							"status": "unhealthy",
						})
					}
				} else {
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer server.Close()

			cfg := &config.Config{
				FastAPIURL: server.URL,
			}

			// AIDEV-NOTE: Create client without health check for this test
			client := &Client{
				httpClient: &http.Client{Timeout: 10 * time.Second},
				baseURL:    server.URL,
				logger:     testLogger,
			}

			ctx := context.Background()
			err := client.HealthCheck(ctx)

			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorMsg)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// AIDEV-NOTE: Test HTTP retry logic
func TestClient_HTTPRetryLogic(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 3 {
			// Fail first 2 attempts
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		// Succeed on 3rd attempt
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]interface{}{"status": "healthy"})
		}
	}))
	defer server.Close()

	cfg := &config.Config{
		FastAPIURL: server.URL,
	}

	client := &Client{
		httpClient: &http.Client{Timeout: 10 * time.Second},
		baseURL:    server.URL,
		logger:     testLogger,
	}

	ctx := context.Background()
	err := client.HealthCheck(ctx)

	// Should succeed after retries
	assert.NoError(t, err)
	assert.Equal(t, 3, attempts, "Expected 3 attempts due to retry logic")
}

// AIDEV-NOTE: Benchmark tests for performance validation
func BenchmarkMatchKOLs(b *testing.B) {
	server := setupTestServer(mockMatchKOLsGraphQLResponse)
	defer server.Close()

	cfg := &config.Config{
		FastAPIURL: server.URL,
	}

	client, err := NewClient(cfg, testLogger)
	require.NoError(b, err)
	defer client.Close()

	request := &MatchKOLsRequest{
		CampaignBrief: "Benchmark test campaign",
		Budget:        5000.0,
		MaxResults:    10,
	}

	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := client.MatchKOLs(ctx, request)
		if err != nil {
			b.Fatalf("MatchKOLs failed: %v", err)
		}
	}
}

func BenchmarkOptimizeBudget(b *testing.B) {
	server := setupTestServer(mockOptimizeBudgetGraphQLResponse)
	defer server.Close()

	cfg := &config.Config{
		FastAPIURL: server.URL,
	}

	client, err := NewClient(cfg, testLogger)
	require.NoError(b, err)
	defer client.Close()

	request := &OptimizeBudgetRequest{
		TotalBudget:   5000.0,
		CampaignGoals: []string{"brand_awareness"},
		KOLCandidates: []string{"kol-123", "kol-456"},
	}

	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := client.OptimizeBudget(ctx, request)
		if err != nil {
			b.Fatalf("OptimizeBudget failed: %v", err)
		}
	}
}

// AIDEV-NOTE: Test connection pooling and concurrency
func TestClient_ConcurrentRequests(b *testing.T) {
	server := setupTestServer(mockMatchKOLsGraphQLResponse)
	defer server.Close()

	cfg := &config.Config{
		FastAPIURL: server.URL,
	}

	client, err := NewClient(cfg, testLogger)
	require.NoError(b, err)
	defer client.Close()

	request := &MatchKOLsRequest{
		CampaignBrief: "Concurrent test campaign",
		Budget:        5000.0,
		MaxResults:    10,
	}

	// AIDEV-NOTE: Test concurrent requests
	concurrency := 20
	done := make(chan error, concurrency)

	for i := 0; i < concurrency; i++ {
		go func() {
			ctx := context.Background()
			_, err := client.MatchKOLs(ctx, request)
			done <- err
		}()
	}

	// AIDEV-NOTE: Collect results
	for i := 0; i < concurrency; i++ {
		select {
		case err := <-done:
			assert.NoError(b, err)
		case <-time.After(10 * time.Second):
			b.Fatal("Timeout waiting for concurrent requests")
		}
	}
}

// AIDEV-NOTE: 250903161030 Test circuit breaker functionality
func TestClient_CircuitBreakerStates(t *testing.T) {
	tests := []struct {
		name            string
		setupServer     func() *httptest.Server
		expectedState   string
		requestCount    int
		expectFallback  bool
		description     string
	}{
		{
			name: "circuit_closed_healthy_requests",
			setupServer: func() *httptest.Server {
				return setupTestServer(mockMatchKOLsGraphQLResponse)
			},
			expectedState: "closed",
			requestCount:  5,
			expectFallback: false,
			description: "Circuit should remain closed with healthy requests",
		},
		{
			name: "circuit_opens_on_failures",
			setupServer: func() *httptest.Server {
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path == "/health" {
						w.WriteHeader(http.StatusOK)
						return
					}
					// Always fail GraphQL requests
					w.WriteHeader(http.StatusInternalServerError)
				}))
			},
			expectedState: "open",
			requestCount:  6, // Exceed failure threshold
			expectFallback: true,
			description: "Circuit should open after consecutive failures",
		},
		{
			name: "circuit_half_open_recovery",
			setupServer: func() *httptest.Server {
				attempts := 0
				return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					if r.URL.Path == "/health" {
						w.WriteHeader(http.StatusOK)
						return
					}
					attempts++
					if attempts <= 5 {
						// Fail first 5 requests to open circuit
						w.WriteHeader(http.StatusInternalServerError)
						return
					}
					// Then succeed to allow recovery
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusOK)
					json.NewEncoder(w).Encode(mockMatchKOLsGraphQLResponse)
				}))
			},
			expectedState: "closed", // Should recover to closed after successful half-open requests
			requestCount:  12, // 5 failures + wait + recovery attempts
			expectFallback: false,
			description: "Circuit should recover from open to closed via half-open state",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := tt.setupServer()
			defer server.Close()

			cfg := &config.Config{
				FastAPIURL: server.URL,
				APITimeout: 5,
			}

			client, err := NewClient(cfg, testLogger)
			require.NoError(t, err)
			defer client.Close()

			request := &MatchKOLsRequest{
				CampaignBrief: "Circuit breaker test",
				Budget:        5000.0,
				MaxResults:    10,
			}

			ctx := context.Background()
			successCount := 0
			fallbackCount := 0

			// Make requests and track circuit state
			for i := 0; i < tt.requestCount; i++ {
				resp, err := client.MatchKOLs(ctx, request)
				
				if err == nil {
					successCount++
					// Check if it's a fallback response
					if resp != nil && len(resp.Matches) == 0 && resp.Meta.AlgorithmUsed == "fallback" {
						fallbackCount++
					}
				}

				// Add small delay for circuit breaker state transitions
				if i < tt.requestCount-1 {
					time.Sleep(100 * time.Millisecond)
				}
			}

			// Wait for circuit recovery if testing half-open transition
			if tt.name == "circuit_half_open_recovery" {
				time.Sleep(2 * time.Second) // Wait for circuit timeout
				// Make additional requests to test recovery
				for i := 0; i < 3; i++ {
					resp, err := client.MatchKOLs(ctx, request)
					if err == nil && resp != nil {
						successCount++
					}
					time.Sleep(100 * time.Millisecond)
				}
			}

			// Validate final circuit state
			metrics := client.GetCircuitBreakerMetrics()
			assert.Equal(t, tt.expectedState, metrics.State, tt.description)

			// Validate fallback behavior
			if tt.expectFallback {
				assert.Greater(t, fallbackCount, 0, "Expected fallback responses when circuit is open")
			} else {
				assert.Equal(t, 0, fallbackCount, "Should not have fallback responses when circuit is healthy")
			}

			t.Logf("Circuit breaker metrics: %+v", metrics)
		})
	}
}

// AIDEV-NOTE: Test circuit breaker exponential backoff
func TestClient_CircuitBreakerExponentialBackoff(t *testing.T) {
	failingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		// Always fail to keep circuit open
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer failingServer.Close()

	cfg := &config.Config{
		FastAPIURL: failingServer.URL,
		APITimeout: 5,
	}

	client, err := NewClient(cfg, testLogger)
	require.NoError(t, err)
	defer client.Close()

	request := &MatchKOLsRequest{
		CampaignBrief: "Backoff test",
		Budget:        5000.0,
		MaxResults:    10,
	}

	ctx := context.Background()

	// Make requests to trigger circuit opening
	for i := 0; i < 6; i++ {
		client.MatchKOLs(ctx, request)
		time.Sleep(100 * time.Millisecond)
	}

	// Get initial metrics
	initialMetrics := client.GetCircuitBreakerMetrics()
	assert.Equal(t, "open", initialMetrics.State)
	initialBackoff := initialMetrics.CurrentBackoff

	// Wait and make more requests to increase backoff
	time.Sleep(1 * time.Second)
	for i := 0; i < 3; i++ {
		client.MatchKOLs(ctx, request)
		time.Sleep(100 * time.Millisecond)
	}

	// Check that backoff has increased
	finalMetrics := client.GetCircuitBreakerMetrics()
	assert.GreaterOrEqual(t, finalMetrics.CurrentBackoff, initialBackoff, "Backoff should increase with failures")
	
	t.Logf("Initial backoff: %v, Final backoff: %v", initialBackoff, finalMetrics.CurrentBackoff)
}

// AIDEV-NOTE: Test circuit breaker metrics and monitoring
func TestClient_CircuitBreakerMetrics(t *testing.T) {
	server := setupTestServer(mockMatchKOLsGraphQLResponse)
	defer server.Close()

	cfg := &config.Config{
		FastAPIURL: server.URL,
		APITimeout: 5,
	}

	client, err := NewClient(cfg, testLogger)
	require.NoError(t, err)
	defer client.Close()

	request := &MatchKOLsRequest{
		CampaignBrief: "Metrics test",
		Budget:        5000.0,
		MaxResults:    10,
	}

	ctx := context.Background()

	// Make some successful requests
	for i := 0; i < 5; i++ {
		_, err := client.MatchKOLs(ctx, request)
		assert.NoError(t, err)
	}

	// Get metrics
	metrics := client.GetCircuitBreakerMetrics()
	stats := client.GetClientStats()

	// Validate metrics structure
	assert.NotNil(t, metrics)
	assert.Equal(t, "fastapi-client", metrics.Name)
	assert.Equal(t, "closed", metrics.State)
	assert.Equal(t, uint64(5), metrics.TotalRequests)
	assert.Equal(t, uint64(5), metrics.TotalSuccesses)
	assert.Equal(t, uint64(0), metrics.TotalFailures)
	assert.Equal(t, 1.0, metrics.SuccessRate)
	assert.Equal(t, 0.0, metrics.FailureRate)
	assert.Equal(t, uint64(0), metrics.CircuitOpenCount)

	// Validate client stats
	assert.NotNil(t, stats)
	assert.Equal(t, uint64(5), stats["total_requests"])
	assert.Equal(t, uint64(5), stats["success_requests"])
	assert.Equal(t, uint64(0), stats["failed_requests"])
	assert.Equal(t, 1.0, stats["success_rate"])

	t.Logf("Circuit breaker metrics: %+v", metrics)
	t.Logf("Client stats: %+v", stats)
}

// AIDEV-NOTE: Test circuit breaker reset functionality
func TestClient_CircuitBreakerReset(t *testing.T) {
	failingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer failingServer.Close()

	cfg := &config.Config{
		FastAPIURL: failingServer.URL,
		APITimeout: 5,
	}

	client, err := NewClient(cfg, testLogger)
	require.NoError(t, err)
	defer client.Close()

	request := &MatchKOLsRequest{
		CampaignBrief: "Reset test",
		Budget:        5000.0,
		MaxResults:    10,
	}

	ctx := context.Background()

	// Make requests to open circuit
	for i := 0; i < 6; i++ {
		client.MatchKOLs(ctx, request)
	}

	// Verify circuit is open
	metrics := client.GetCircuitBreakerMetrics()
	assert.Equal(t, "open", metrics.State)
	assert.Greater(t, metrics.TotalFailures, uint64(0))

	// Reset circuit breaker
	client.ResetCircuitBreaker()

	// Verify reset
	resetMetrics := client.GetCircuitBreakerMetrics()
	assert.Equal(t, "closed", resetMetrics.State)
	assert.Equal(t, uint64(0), resetMetrics.TotalRequests)
	assert.Equal(t, uint64(0), resetMetrics.TotalFailures)
	assert.Equal(t, uint64(0), resetMetrics.TotalSuccesses)
	assert.Equal(t, uint64(0), resetMetrics.CircuitOpenCount)

	stats := client.GetClientStats()
	assert.Equal(t, uint64(0), stats["total_requests"])
	assert.Equal(t, uint64(0), stats["failed_requests"])
	assert.Equal(t, uint64(0), stats["success_requests"])
}

// AIDEV-NOTE: Test error handling with various HTTP status codes
func TestClient_ErrorHandling(t *testing.T) {
	tests := []struct {
		name         string
		serverStatus int
		serverBody   string
		expectedErr  string
	}{
		{
			name:         "400_bad_request",
			serverStatus: http.StatusBadRequest,
			serverBody:   `{"code": 400, "message": "Invalid request"}`,
			expectedErr:  "FastAPI error 400: Invalid request",
		},
		{
			name:         "401_unauthorized",
			serverStatus: http.StatusUnauthorized,
			serverBody:   `{"code": 401, "message": "Unauthorized"}`,
			expectedErr:  "FastAPI error 401: Unauthorized",
		},
		{
			name:         "500_internal_error",
			serverStatus: http.StatusInternalServerError,
			serverBody:   "Internal server error",
			expectedErr:  "HTTP 500: Internal server error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.serverStatus)
				w.Write([]byte(tt.serverBody))
			}))
			defer server.Close()

			client := &Client{
				httpClient: &http.Client{Timeout: 10 * time.Second},
				baseURL:    server.URL,
				logger:     testLogger,
			}

			ctx := context.Background()
			_, err := client.makeHTTPRequest(ctx, "GET", "/test", nil)

			assert.Error(t, err)
			assert.Contains(t, err.Error(), tt.expectedErr)
		})
	}
}

// AIDEV-NOTE: Benchmark circuit breaker performance impact
func BenchmarkCircuitBreakerOverhead(b *testing.B) {
	server := setupTestServer(mockMatchKOLsGraphQLResponse)
	defer server.Close()

	cfg := &config.Config{
		FastAPIURL: server.URL,
		APITimeout: 30,
	}

	client, err := NewClient(cfg, testLogger)
	require.NoError(b, err)
	defer client.Close()

	request := &MatchKOLsRequest{
		CampaignBrief: "Benchmark circuit breaker overhead",
		Budget:        5000.0,
		MaxResults:    10,
	}

	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := client.MatchKOLs(ctx, request)
		if err != nil {
			b.Fatalf("MatchKOLs failed: %v", err)
		}
	}
}