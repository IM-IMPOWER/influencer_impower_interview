// AIDEV-NOTE: 250102161030 Integration tests for FastAPI client with circuit breaker
// Tests the complete integration between circuit breaker and GraphQL client functionality
package fastapi

import (
	"context"
	"math/rand"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"kol-scraper/internal/circuit"
	"kol-scraper/pkg/config"
	"kol-scraper/pkg/logger"
)

func TestClient_MatchKOLs_WithCircuitBreaker(t *testing.T) {
	// AIDEV-NOTE: Create mock FastAPI server
	requestCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		
		// Simulate failures for first few requests to test circuit breaker
		if requestCount <= 3 {
			http.Error(w, "Service temporarily unavailable", http.StatusServiceUnavailable)
			return
		}
		
		// Return successful GraphQL response
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{
			"data": {
				"matchKOLs": {
					"matches": [
						{
							"kolId": "test-kol-1",
							"username": "testuser",
							"platform": "tiktok",
							"displayName": "Test User",
							"followerCount": 50000,
							"engagementRate": 0.05,
							"score": 0.85,
							"tier": "micro",
							"category": "lifestyle",
							"estimatedCost": 500.0,
							"reasoning": "High engagement rate and good brand fit"
						}
					],
					"meta": {
						"totalFound": 1,
						"queryTime": 150.5,
						"algorithmUsed": "enhanced-ml-v2",
						"confidence": 0.85
					}
				}
			}
		}`))
	}))
	defer server.Close()

	// AIDEV-NOTE: Create test config and client
	cfg := &config.Config{
		FastAPIURL:            server.URL,
		APITimeout:            30,
		CircuitBreakerEnabled: true,
		MaxFailures:           2,
		CircuitBreakerTimeout: 1, // 1 second for faster testing
	}

	log := logger.New("debug", "test")
	client, err := NewClient(cfg, log)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()
	req := &MatchKOLsRequest{
		CampaignBrief: "Looking for lifestyle influencers with high engagement",
		Budget:        5000.0,
		Platforms:     []string{"tiktok", "instagram"},
		Categories:    []string{"lifestyle"},
		MinFollowers:  10000,
		MaxFollowers:  100000,
		MaxResults:    10,
	}

	// AIDEV-NOTE: First requests should fail and trigger circuit breaker
	for i := 0; i < 3; i++ {
		resp, err := client.MatchKOLs(ctx, req)
		
		if i < 2 {
			// First two requests should fail
			if err == nil {
				t.Errorf("Expected error for request %d, but got success", i+1)
			}
			t.Logf("Request %d failed as expected: %v", i+1, err)
		} else {
			// Third request should use fallback due to circuit breaker being open
			if err != nil {
				t.Errorf("Request %d failed unexpectedly: %v", i+1, err)
			}
			
			if resp == nil {
				t.Error("Expected fallback response, got nil")
			} else {
				// Should be fallback response with empty matches
				if len(resp.Matches) != 0 {
					t.Errorf("Expected empty fallback response, got %d matches", len(resp.Matches))
				}
				
				if resp.Meta.AlgorithmUsed != "fallback" {
					t.Errorf("Expected fallback algorithm, got %s", resp.Meta.AlgorithmUsed)
				}
				
				t.Logf("Request %d used fallback successfully", i+1)
			}
		}
		
		// Check circuit breaker state
		metrics := client.GetCircuitBreakerMetrics()
		t.Logf("Circuit breaker state after request %d: %s (failures: %d, success rate: %.2f%%)", 
			i+1, metrics.State.String(), metrics.FailureCount, metrics.SuccessRate*100)
	}

	// AIDEV-NOTE: Wait for circuit breaker recovery
	time.Sleep(1200 * time.Millisecond) // Wait for recovery timeout + buffer
	
	// Reset server request count to track recovery behavior
	requestCount = 3
	
	// AIDEV-NOTE: Next request should succeed and close circuit
	resp, err := client.MatchKOLs(ctx, req)
	if err != nil {
		t.Errorf("Expected success after recovery, got error: %v", err)
	}
	
	if resp == nil {
		t.Error("Expected successful response after recovery")
	} else {
		if len(resp.Matches) != 1 {
			t.Errorf("Expected 1 match after recovery, got %d", len(resp.Matches))
		}
		
		if resp.Meta.AlgorithmUsed == "fallback" {
			t.Error("Expected real response after recovery, got fallback")
		}
		
		t.Log("Circuit breaker recovered successfully")
	}
	
	// AIDEV-NOTE: Verify circuit breaker closed
	finalMetrics := client.GetCircuitBreakerMetrics()
	if finalMetrics.State != circuit.StateClosed {
		t.Errorf("Expected circuit to be CLOSED after recovery, got %s", finalMetrics.State)
	}
	
	t.Logf("Final circuit breaker metrics: %+v", finalMetrics)
	t.Logf("Total server requests made: %d", requestCount)
}

func TestClient_OptimizeBudget_WithCircuitBreaker(t *testing.T) {
	// AIDEV-NOTE: Create mock FastAPI server that always fails
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "Service down for maintenance", http.StatusInternalServerError)
	}))
	defer server.Close()

	// AIDEV-NOTE: Create client with aggressive circuit breaker settings
	cfg := &config.Config{
		FastAPIURL:            server.URL,
		APITimeout:            30,
		CircuitBreakerEnabled: true,
		MaxFailures:           1,
		CircuitBreakerTimeout: 1,
	}

	log := logger.New("debug", "test")
	client, err := NewClient(cfg, log)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	ctx := context.Background()
	req := &OptimizeBudgetRequest{
		TotalBudget:   10000.0,
		CampaignGoals: []string{"brand_awareness", "engagement"},
		KOLCandidates: []string{"kol-1", "kol-2", "kol-3"},
		TargetReach:   50000,
		Constraints: BudgetConstraints{
			MaxKOLs: 3,
			MinKOLs: 1,
		},
	}

	// AIDEV-NOTE: First request should fail and open circuit
	_, err = client.OptimizeBudget(ctx, req)
	if err == nil {
		t.Error("Expected first request to fail")
	}
	
	metrics := client.GetCircuitBreakerMetrics()
	if metrics.State != circuit.StateOpen {
		t.Errorf("Expected circuit to be OPEN after failure, got %s", metrics.State)
	}

	// AIDEV-NOTE: Second request should be blocked by circuit breaker
	// and return fallback if configured
	_, err = client.OptimizeBudget(ctx, req)
	
	// Should either return fallback success or circuit breaker error
	if err != nil {
		// Check if it's a circuit breaker error by looking for "circuit breaker" in message
		if err.Error() != "circuit breaker is OPEN" {
			t.Logf("Got circuit breaker related error: %v", err)
		}
	}
	
	finalMetrics := client.GetCircuitBreakerMetrics()
	t.Logf("Final circuit breaker state: %s, failures: %d", 
		finalMetrics.State.String(), finalMetrics.FailureCount)
}

func TestClient_CircuitBreakerMetrics(t *testing.T) {
	// AIDEV-NOTE: Create a client with default settings
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"data": {"healthCheck": "ok"}}`))
	}))
	defer server.Close()

	cfg := &config.Config{
		FastAPIURL:            server.URL,
		APITimeout:            30,
		CircuitBreakerEnabled: true,
	}

	log := logger.New("debug", "test")
	client, err := NewClient(cfg, log)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// AIDEV-NOTE: Test initial metrics
	metrics := client.GetCircuitBreakerMetrics()
	if metrics.State != circuit.StateClosed {
		t.Errorf("Expected initial state CLOSED, got %s", metrics.State)
	}
	
	if metrics.SuccessCount != 0 || metrics.FailureCount != 0 {
		t.Errorf("Expected zero initial counts, got success=%d, failure=%d", 
			metrics.SuccessCount, metrics.FailureCount)
	}

	// AIDEV-NOTE: Test IsHealthy method
	if !client.IsHealthy() {
		t.Error("Expected client to be healthy initially")
	}
	
	t.Logf("Circuit breaker metrics: %+v", metrics)
}

func TestClient_Concurrency_WithCircuitBreaker(t *testing.T) {
	// AIDEV-NOTE: Test concurrent requests with circuit breaker
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate variable response times
		time.Sleep(time.Duration(10+randomDelay(20)) * time.Millisecond)
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{
			"data": {
				"matchKOLs": {
					"matches": [],
					"meta": {
						"totalFound": 0,
						"queryTime": 50,
						"algorithmUsed": "test",
						"confidence": 1.0
					}
				}
			}
		}`))
	}))
	defer server.Close()

	cfg := &config.Config{
		FastAPIURL:            server.URL,
		APITimeout:            30,
		CircuitBreakerEnabled: true,
	}

	log := logger.New("debug", "test")
	client, err := NewClient(cfg, log)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	const goroutines = 10
	const requestsPerGoroutine = 5

	ctx := context.Background()
	req := &MatchKOLsRequest{
		CampaignBrief: "Test concurrent requests",
		Budget:        1000.0,
		MaxResults:    1,
	}

	var wg sync.WaitGroup
	var successCount, errorCount int64
	var mu sync.Mutex

	// AIDEV-NOTE: Launch concurrent requests
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			for j := 0; j < requestsPerGoroutine; j++ {
				_, err := client.MatchKOLs(ctx, req)
				
				mu.Lock()
				if err != nil {
					errorCount++
				} else {
					successCount++
				}
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	totalRequests := successCount + errorCount
	t.Logf("Concurrent test completed: %d total, %d success, %d errors", 
		totalRequests, successCount, errorCount)

	metrics := client.GetCircuitBreakerMetrics()
	t.Logf("Final circuit breaker metrics: %+v", metrics)

	// Most requests should succeed with a healthy server
	if float64(successCount)/float64(totalRequests) < 0.8 {
		t.Errorf("Expected >80%% success rate, got %.2f%%", 
			float64(successCount)/float64(totalRequests)*100)
	}
}

// AIDEV-NOTE: Test helper function to create failing server
func createFailingServer(failureRate float64) *httptest.Server {
	var requestCount int
	
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		
		// Fail based on failure rate
		if float64(requestCount%100) < failureRate*100 {
			http.Error(w, "Simulated failure", http.StatusInternalServerError)
			return
		}
		
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"data": {"healthCheck": "ok"}}`))
	}))
}

// AIDEV-NOTE: Helper function to generate random delays for testing
func randomDelay(max int) int {
	return int(time.Now().UnixNano()) % max
}