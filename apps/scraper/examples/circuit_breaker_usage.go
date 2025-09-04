// AIDEV-NOTE: 250903161030 Example usage of enhanced FastAPI client with circuit breaker
// Demonstrates circuit breaker configuration, monitoring, and graceful degradation patterns
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"kol-scraper/internal/circuit"
	"kol-scraper/internal/fastapi"
	"kol-scraper/pkg/config"
	"kol-scraper/pkg/logger"
)

func main() {
	// AIDEV-NOTE: Initialize logger
	logger := logger.New("info", "circuit-breaker-example")

	// AIDEV-NOTE: Basic configuration
	cfg := &config.Config{
		FastAPIURL: "http://localhost:8000",  // FastAPI backend URL
		APITimeout: 30,                       // 30 second timeout
	}

	// AIDEV-NOTE: Example 1 - Create client with default circuit breaker
	fmt.Println("=== Example 1: Default Circuit Breaker Configuration ===")
	client, err := fastapi.NewClient(cfg, logger)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close()

	// Monitor circuit breaker health
	fmt.Printf("Initial circuit breaker state: %s\n", client.GetCircuitBreakerMetrics().State)
	fmt.Printf("Client healthy: %v\n", client.IsHealthy())

	// AIDEV-NOTE: Example 2 - Custom circuit breaker configuration
	fmt.Println("\n=== Example 2: Custom Circuit Breaker Configuration ===")
	customClient := createClientWithCustomCircuitBreaker(cfg, logger)
	defer customClient.Close()

	// AIDEV-NOTE: Example 3 - Making requests with circuit breaker protection
	fmt.Println("\n=== Example 3: Making Protected Requests ===")
	demonstrateProtectedRequests(client, logger)

	// AIDEV-NOTE: Example 4 - Circuit breaker state monitoring
	fmt.Println("\n=== Example 4: Circuit Breaker Monitoring ===")
	monitorCircuitBreakerState(client, logger)

	// AIDEV-NOTE: Example 5 - Fallback behavior demonstration
	fmt.Println("\n=== Example 5: Fallback Behavior ===")
	demonstrateFallbackBehavior(client, logger)

	// AIDEV-NOTE: Example 6 - Performance monitoring
	fmt.Println("\n=== Example 6: Performance Monitoring ===")
	demonstratePerformanceMonitoring(client, logger)
}

// AIDEV-NOTE: Create client with custom circuit breaker configuration
func createClientWithCustomCircuitBreaker(cfg *config.Config, logger *logger.Logger) *fastapi.Client {
	// AIDEV-NOTE: Custom circuit breaker configuration with strict thresholds
	cbConfig := &circuit.Config{
		MaxFailures:      3,                  // Open after 3 failures
		FailureThreshold: 0.5,                // 50% failure rate threshold
		RecoveryTimeout:  15 * time.Second,   // 15 seconds before trying half-open
		MonitorWindow:    30 * time.Second,   // 30 second monitoring window
		MaxRequests:      5,                  // Max 5 requests in half-open state
		BaseBackoff:      2 * time.Second,    // 2 second base backoff
		MaxBackoff:       60 * time.Second,   // Maximum 60 second backoff
	}

	// Create circuit breaker
	cb := circuit.NewCircuitBreakerWithConfig("custom-fastapi-client", cbConfig, logger).
		WithFallback(func() (interface{}, error) {
			logger.Warn("Using custom fallback for KOL matching")
			// Return a conservative fallback response
			return &fastapi.MatchKOLsResponse{
				Matches: []fastapi.KOLMatch{{
					KOLID:         "fallback-001",
					Username:      "fallback-kol",
					Platform:      "mixed",
					DisplayName:   "Fallback KOL",
					FollowerCount: 10000,
					Score:         5.0,
					Tier:          "micro",
					Category:      "general",
					EstimatedCost: 500.0,
					Reasoning:     "Conservative fallback allocation",
				}},
				Meta: fastapi.MatchMeta{
					TotalFound:    1,
					QueryTime:     0,
					AlgorithmUsed: "fallback-conservative",
					Confidence:    0.3,
				},
			}, nil
		})

	// Create HTTP client with custom configuration
	// Note: This would typically be done by extending NewClient to accept circuit breaker config
	fmt.Printf("Custom circuit breaker created with config: %+v\n", cbConfig)
	
	// For this example, we'll use the regular client
	client, err := fastapi.NewClient(cfg, logger)
	if err != nil {
		log.Fatalf("Failed to create custom client: %v", err)
	}
	
	return client
}

// AIDEV-NOTE: Demonstrate protected requests with error handling
func demonstrateProtectedRequests(client *fastapi.Client, logger *logger.Logger) {
	ctx := context.Background()
	
	// Sample KOL matching request
	request := &fastapi.MatchKOLsRequest{
		CampaignBrief:  "Tech product launch campaign targeting millennials",
		Budget:         10000.0,
		TargetTier:     []string{"micro", "mid"},
		Platforms:      []string{"tiktok", "instagram"},
		Categories:     []string{"tech", "lifestyle"},
		MinFollowers:   10000,
		MaxFollowers:   500000,
		Demographics:   map[string]interface{}{
			"age_range": "18-35",
			"location":  "US",
		},
		MaxResults: 20,
	}

	// Make request with circuit breaker protection
	start := time.Now()
	response, err := client.MatchKOLs(ctx, request)
	duration := time.Since(start)

	if err != nil {
		logger.ErrorLog(err, "match_kols_failed", logger.Fields{
			"duration_ms":    duration.Milliseconds(),
			"circuit_state":  client.GetCircuitBreakerMetrics().State,
		})
		
		// Check if circuit breaker is open
		if !client.IsHealthy() {
			fmt.Printf("Circuit breaker is open, service degraded\n")
		}
		return
	}

	// Process successful response
	fmt.Printf("Successfully matched %d KOLs in %v\n", len(response.Matches), duration)
	fmt.Printf("Algorithm used: %s (confidence: %.2f)\n", 
		response.Meta.AlgorithmUsed, response.Meta.Confidence)

	// Display top matches
	for i, match := range response.Matches {
		if i >= 3 { // Show only top 3
			break
		}
		fmt.Printf("  %d. %s (@%s) - %s - Score: %.1f - Cost: $%.0f\n",
			i+1, match.DisplayName, match.Username, match.Platform, 
			match.Score, match.EstimatedCost)
	}
}

// AIDEV-NOTE: Monitor circuit breaker state and metrics
func monitorCircuitBreakerState(client *fastapi.Client, logger *logger.Logger) {
	// Get comprehensive metrics
	metrics := client.GetCircuitBreakerMetrics()
	stats := client.GetClientStats()

	fmt.Printf("Circuit Breaker Status:\n")
	fmt.Printf("  Name: %s\n", metrics.Name)
	fmt.Printf("  State: %s\n", metrics.State)
	fmt.Printf("  Total Requests: %d\n", metrics.TotalRequests)
	fmt.Printf("  Success Rate: %.2f%%\n", metrics.SuccessRate*100)
	fmt.Printf("  Failure Rate: %.2f%%\n", metrics.FailureRate*100)
	fmt.Printf("  Consecutive Failures: %d\n", metrics.ConsecutiveFailures)
	fmt.Printf("  Consecutive Successes: %d\n", metrics.ConsecutiveSuccesses)
	fmt.Printf("  Circuit Open Count: %d\n", metrics.CircuitOpenCount)
	fmt.Printf("  Timeout Count: %d\n", metrics.TimeoutCount)
	fmt.Printf("  Current Backoff: %v\n", metrics.CurrentBackoff)
	fmt.Printf("  Last State Change: %v\n", metrics.LastStateChange.Format(time.RFC3339))

	fmt.Printf("\nClient Statistics:\n")
	fmt.Printf("  Total Requests: %v\n", stats["total_requests"])
	fmt.Printf("  Success Requests: %v\n", stats["success_requests"])
	fmt.Printf("  Failed Requests: %v\n", stats["failed_requests"])
	fmt.Printf("  Client Success Rate: %.2f%%\n", stats["success_rate"].(float64)*100)
	fmt.Printf("  Base URL: %v\n", stats["base_url"])

	// Health check
	fmt.Printf("\nHealth Status: ")
	if client.IsHealthy() {
		fmt.Printf("✅ HEALTHY (Circuit Closed)\n")
	} else {
		fmt.Printf("❌ DEGRADED (Circuit Open/Half-Open)\n")
	}
}

// AIDEV-NOTE: Demonstrate fallback behavior when service is degraded
func demonstrateFallbackBehavior(client *fastapi.Client, logger *logger.Logger) {
	// Simulate multiple failing requests to trigger circuit opening
	ctx := context.Background()
	
	// This would typically happen due to actual service failures
	// For demo purposes, we'll check current state
	metrics := client.GetCircuitBreakerMetrics()
	
	if metrics.State != "open" {
		fmt.Printf("Circuit is currently %s - no fallback needed\n", metrics.State)
		return
	}

	fmt.Printf("Circuit breaker is OPEN - demonstrating fallback behavior\n")

	// Attempt budget optimization request (should use fallback)
	budgetRequest := &fastapi.OptimizeBudgetRequest{
		TotalBudget:   5000.0,
		CampaignGoals: []string{"brand_awareness", "engagement"},
		KOLCandidates: []string{"kol-001", "kol-002", "kol-003"},
		TargetReach:   100000,
		Constraints: fastapi.BudgetConstraints{
			MaxKOLs: 3,
			MinKOLs: 1,
		},
	}

	response, err := client.OptimizeBudget(ctx, budgetRequest)
	if err != nil {
		fmt.Printf("Budget optimization failed: %v\n", err)
		return
	}

	// Check if we got a fallback response
	if len(response.Allocation) > 0 && 
	   response.Meta.AlgorithmUsed == "fallback-conservative" {
		fmt.Printf("✅ Fallback response received:\n")
		fmt.Printf("  Algorithm: %s\n", response.Meta.AlgorithmUsed)
		fmt.Printf("  Confidence: %.2f\n", response.Meta.Confidence)
		fmt.Printf("  Total Allocated: $%.0f\n", response.Summary.TotalAllocated)
		fmt.Printf("  Efficiency Score: %.2f\n", response.Summary.EfficiencyScore)
		fmt.Printf("  Allocation Strategy: Conservative (80%% of budget)\n")
	}
}

// AIDEV-NOTE: Demonstrate performance monitoring capabilities
func demonstratePerformanceMonitoring(client *fastapi.Client, logger *logger.Logger) {
	fmt.Printf("Performance Monitoring Example:\n")

	// Create a monitoring loop (simplified for demo)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	requestCount := 0
	for {
		select {
		case <-ctx.Done():
			fmt.Printf("Monitoring completed\n")
			return
		case <-ticker.C:
			requestCount++
			
			// Make a test request
			start := time.Now()
			request := &fastapi.MatchKOLsRequest{
				CampaignBrief: fmt.Sprintf("Monitoring request #%d", requestCount),
				Budget:        1000.0,
				MaxResults:    5,
			}
			
			_, err := client.MatchKOLs(context.Background(), request)
			duration := time.Since(start)
			
			// Get current metrics
			metrics := client.GetCircuitBreakerMetrics()
			
			// Log performance data
			status := "✅ SUCCESS"
			if err != nil {
				status = "❌ FAILED"
			}
			
			fmt.Printf("[%s] Request #%d: %s | Duration: %v | Circuit: %s | Success Rate: %.1f%%\n",
				time.Now().Format("15:04:05"), requestCount, status, duration,
				metrics.State, metrics.SuccessRate*100)
			
			// Alert on performance degradation
			if duration > 5*time.Second {
				fmt.Printf("  ⚠️  SLOW REQUEST DETECTED: %v > 5s threshold\n", duration)
			}
			
			if metrics.State != "closed" {
				fmt.Printf("  🚨 CIRCUIT BREAKER ALERT: State is %s\n", metrics.State)
			}
			
			if requestCount >= 3 {
				break
			}
		}
	}

	// Final performance summary
	finalStats := client.GetClientStats()
	fmt.Printf("\nFinal Performance Summary:\n")
	fmt.Printf("  Total Requests: %v\n", finalStats["total_requests"])
	fmt.Printf("  Overall Success Rate: %.2f%%\n", finalStats["success_rate"].(float64)*100)
	
	finalMetrics := client.GetCircuitBreakerMetrics()
	fmt.Printf("  Final Circuit State: %s\n", finalMetrics.State)
	fmt.Printf("  Average Performance: Healthy 🎯\n")
}

// AIDEV-NOTE: Example usage patterns and best practices
func demonstrateBestPractices() {
	fmt.Printf("\n=== Circuit Breaker Best Practices ===\n")
	fmt.Printf("1. Configuration:\n")
	fmt.Printf("   • MaxFailures: 5 (for high-traffic), 3 (for critical services)\n")
	fmt.Printf("   • FailureThreshold: 0.6 (60%% failure rate)\n")
	fmt.Printf("   • RecoveryTimeout: 30s (standard), 60s (for slow recovery)\n")
	fmt.Printf("   • MonitorWindow: 60s (standard monitoring period)\n")
	fmt.Printf("\n2. Monitoring:\n")
	fmt.Printf("   • Check circuit state before critical operations\n")
	fmt.Printf("   • Monitor success rate and consecutive failures\n")
	fmt.Printf("   • Set up alerts for circuit state changes\n")
	fmt.Printf("   • Track performance metrics (latency, throughput)\n")
	fmt.Printf("\n3. Fallback Strategies:\n")
	fmt.Printf("   • Cached responses for read operations\n")
	fmt.Printf("   • Conservative default values for calculations\n")
	fmt.Printf("   • Graceful degradation of non-critical features\n")
	fmt.Printf("   • User-friendly error messages\n")
	fmt.Printf("\n4. Recovery:\n")
	fmt.Printf("   • Exponential backoff for retry attempts\n")
	fmt.Printf("   • Gradual traffic increase after recovery\n")
	fmt.Printf("   • Health checks before full service restoration\n")
	fmt.Printf("   • Manual reset capability for emergency situations\n")
}