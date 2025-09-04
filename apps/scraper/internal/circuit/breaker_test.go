// AIDEV-NOTE: 250102161030 Comprehensive tests for circuit breaker implementation
// Tests all state transitions, failure detection, recovery, and performance aspects
package circuit

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestCircuitBreaker_StateTransitions(t *testing.T) {
	config := Config{
		MaxFailures:      3,
		FailureThreshold: 0.6,
		RecoveryTimeout:  100 * time.Millisecond,
		MonitorWindow:    1 * time.Second,
		BaseBackoff:      50 * time.Millisecond,
		MaxBackoff:       200 * time.Millisecond,
		HalfOpenRequests: 2,
	}
	
	cb := NewCircuitBreaker(config)
	ctx := context.Background()

	// AIDEV-NOTE: Test initial state is CLOSED
	if cb.GetState() != StateClosed {
		t.Errorf("Expected initial state CLOSED, got %s", cb.GetState())
	}

	// AIDEV-NOTE: Test successful requests keep circuit CLOSED
	for i := 0; i < 5; i++ {
		_, err := cb.Execute(ctx, func() (interface{}, error) {
			return "success", nil
		})
		if err != nil {
			t.Errorf("Unexpected error on success: %v", err)
		}
		if cb.GetState() != StateClosed {
			t.Errorf("Expected CLOSED after success %d, got %s", i+1, cb.GetState())
		}
	}

	// AIDEV-NOTE: Test failures transition to OPEN
	for i := 0; i < config.MaxFailures; i++ {
		cb.Execute(ctx, func() (interface{}, error) {
			return nil, errors.New("test failure")
		})
	}
	
	if cb.GetState() != StateOpen {
		t.Errorf("Expected OPEN after %d failures, got %s", config.MaxFailures, cb.GetState())
	}

	// AIDEV-NOTE: Test requests are blocked in OPEN state
	_, err := cb.Execute(ctx, func() (interface{}, error) {
		t.Error("Function should not be called in OPEN state")
		return nil, nil
	})
	if err == nil {
		t.Error("Expected error in OPEN state")
	}

	// AIDEV-NOTE: Test transition to HALF_OPEN after recovery timeout
	time.Sleep(config.RecoveryTimeout + 10*time.Millisecond)
	
	_, err = cb.Execute(ctx, func() (interface{}, error) {
		return "test", nil
	})
	if err != nil {
		t.Errorf("Expected success in HALF_OPEN, got error: %v", err)
	}
	
	if cb.GetState() != StateHalfOpen {
		t.Errorf("Expected HALF_OPEN after recovery, got %s", cb.GetState())
	}

	// AIDEV-NOTE: Test successful requests in HALF_OPEN transition back to CLOSED
	_, err = cb.Execute(ctx, func() (interface{}, error) {
		return "success", nil
	})
	if err != nil {
		t.Errorf("Unexpected error in HALF_OPEN: %v", err)
	}
	
	if cb.GetState() != StateClosed {
		t.Errorf("Expected CLOSED after successful half-open requests, got %s", cb.GetState())
	}
}

func TestCircuitBreaker_FailureThreshold(t *testing.T) {
	config := Config{
		MaxFailures:      10, // High number so threshold triggers first
		FailureThreshold: 0.5, // 50% failure rate
		RecoveryTimeout:  100 * time.Millisecond,
		MonitorWindow:    1 * time.Second,
		BaseBackoff:      50 * time.Millisecond,
		MaxBackoff:       200 * time.Millisecond,
		HalfOpenRequests: 2,
	}
	
	cb := NewCircuitBreaker(config)
	ctx := context.Background()

	// AIDEV-NOTE: Create mixed success/failure requests to hit threshold
	successCount := 0
	failureCount := 0
	
	for i := 0; i < 10; i++ {
		shouldFail := i%2 == 0 // 50% failure rate
		
		cb.Execute(ctx, func() (interface{}, error) {
			if shouldFail {
				failureCount++
				return nil, errors.New("test failure")
			}
			successCount++
			return "success", nil
		})
	}

	// AIDEV-NOTE: Circuit should be OPEN due to failure threshold
	if cb.GetState() != StateOpen {
		t.Errorf("Expected OPEN due to failure threshold, got %s", cb.GetState())
	}
	
	metrics := cb.GetMetrics()
	if metrics.SuccessRate > 0.5 {
		t.Errorf("Expected success rate <= 0.5, got %f", metrics.SuccessRate)
	}
}

func TestCircuitBreaker_ExponentialBackoff(t *testing.T) {
	config := Config{
		MaxFailures:      2,
		FailureThreshold: 0.8,
		RecoveryTimeout:  50 * time.Millisecond,
		MonitorWindow:    1 * time.Second,
		BaseBackoff:      10 * time.Millisecond,
		MaxBackoff:       100 * time.Millisecond,
		HalfOpenRequests: 1,
	}
	
	cb := NewCircuitBreaker(config)
	ctx := context.Background()

	initialBackoff := cb.GetMetrics().CurrentBackoff
	
	// AIDEV-NOTE: Trigger failures to test backoff progression
	for i := 0; i < 3; i++ {
		cb.Execute(ctx, func() (interface{}, error) {
			return nil, errors.New("test failure")
		})
		
		metrics := cb.GetMetrics()
		if i > 0 && metrics.CurrentBackoff <= initialBackoff {
			t.Errorf("Expected backoff to increase, got %v after %v", 
				metrics.CurrentBackoff, initialBackoff)
		}
		
		if metrics.CurrentBackoff > config.MaxBackoff {
			t.Errorf("Backoff exceeded maximum: %v > %v", 
				metrics.CurrentBackoff, config.MaxBackoff)
		}
		
		initialBackoff = metrics.CurrentBackoff
	}
}

func TestCircuitBreaker_ConcurrencySafety(t *testing.T) {
	config := DefaultConfig()
	cb := NewCircuitBreaker(config)
	ctx := context.Background()
	
	const goroutines = 100
	const requestsPerGoroutine = 50
	
	var wg sync.WaitGroup
	var successCount, failureCount int64
	var mu sync.Mutex
	
	// AIDEV-NOTE: Test concurrent requests
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			
			for j := 0; j < requestsPerGoroutine; j++ {
				shouldFail := (id+j)%3 == 0 // ~33% failure rate
				
				_, err := cb.Execute(ctx, func() (interface{}, error) {
					// Simulate some work
					time.Sleep(1 * time.Millisecond)
					
					if shouldFail {
						return nil, errors.New("test failure")
					}
					return "success", nil
				})
				
				mu.Lock()
				if err != nil {
					failureCount++
				} else {
					successCount++
				}
				mu.Unlock()
			}
		}(i)
	}
	
	wg.Wait()
	
	metrics := cb.GetMetrics()
	totalRequests := successCount + failureCount
	
	t.Logf("Concurrent test completed: %d total requests, %d successes, %d failures", 
		totalRequests, successCount, failureCount)
	t.Logf("Circuit breaker state: %s, metrics: %+v", cb.GetState(), metrics)
	
	// AIDEV-NOTE: Verify metrics consistency
	if metrics.TotalRequests != totalRequests {
		t.Errorf("Metrics mismatch: expected %d total requests, got %d", 
			totalRequests, metrics.TotalRequests)
	}
}

func TestCircuitBreaker_FallbackFunction(t *testing.T) {
	config := Config{
		MaxFailures:      1,
		FailureThreshold: 0.8,
		RecoveryTimeout:  100 * time.Millisecond,
		MonitorWindow:    1 * time.Second,
		BaseBackoff:      50 * time.Millisecond,
		MaxBackoff:       200 * time.Millisecond,
		HalfOpenRequests: 1,
	}
	
	cb := NewCircuitBreaker(config)
	ctx := context.Background()
	
	fallbackCalled := false
	cb.SetFallback(func(err error) (interface{}, error) {
		fallbackCalled = true
		return "fallback response", nil
	})
	
	// AIDEV-NOTE: Trigger failure to open circuit
	cb.Execute(ctx, func() (interface{}, error) {
		return nil, errors.New("test failure")
	})
	
	// AIDEV-NOTE: Test fallback is called when circuit is open
	result, err := cb.Execute(ctx, func() (interface{}, error) {
		t.Error("Main function should not be called when circuit is open")
		return nil, nil
	})
	
	if !fallbackCalled {
		t.Error("Expected fallback function to be called")
	}
	
	if err != nil {
		t.Errorf("Expected no error with fallback, got: %v", err)
	}
	
	if result != "fallback response" {
		t.Errorf("Expected fallback response, got: %v", result)
	}
}

func TestCircuitBreaker_StateChangeCallback(t *testing.T) {
	config := Config{
		MaxFailures:      2,
		FailureThreshold: 0.8,
		RecoveryTimeout:  50 * time.Millisecond,
		MonitorWindow:    1 * time.Second,
		BaseBackoff:      25 * time.Millisecond,
		MaxBackoff:       100 * time.Millisecond,
		HalfOpenRequests: 1,
	}
	
	cb := NewCircuitBreaker(config)
	ctx := context.Background()
	
	var stateChanges []string
	var mu sync.Mutex
	
	cb.SetOnStateChange(func(from, to State) {
		mu.Lock()
		stateChanges = append(stateChanges, fmt.Sprintf("%s->%s", from, to))
		mu.Unlock()
	})
	
	// AIDEV-NOTE: Trigger state transitions
	// CLOSED -> OPEN
	for i := 0; i < config.MaxFailures; i++ {
		cb.Execute(ctx, func() (interface{}, error) {
			return nil, errors.New("test failure")
		})
	}
	
	// Wait for recovery and trigger OPEN -> HALF_OPEN
	time.Sleep(config.RecoveryTimeout + 10*time.Millisecond)
	cb.Execute(ctx, func() (interface{}, error) {
		return "success", nil
	})
	
	// Wait for callback to complete
	time.Sleep(10 * time.Millisecond)
	
	mu.Lock()
	defer mu.Unlock()
	
	expectedTransitions := []string{"CLOSED->OPEN", "OPEN->HALF_OPEN", "HALF_OPEN->CLOSED"}
	
	if len(stateChanges) != len(expectedTransitions) {
		t.Errorf("Expected %d state changes, got %d: %v", 
			len(expectedTransitions), len(stateChanges), stateChanges)
		return
	}
	
	for i, expected := range expectedTransitions {
		if i < len(stateChanges) && stateChanges[i] != expected {
			t.Errorf("Expected state change %d to be %s, got %s", 
				i, expected, stateChanges[i])
		}
	}
}

func TestCircuitBreaker_Reset(t *testing.T) {
	config := DefaultConfig()
	cb := NewCircuitBreaker(config)
	ctx := context.Background()
	
	// AIDEV-NOTE: Trigger failures to change state
	for i := 0; i < config.MaxFailures; i++ {
		cb.Execute(ctx, func() (interface{}, error) {
			return nil, errors.New("test failure")
		})
	}
	
	if cb.GetState() != StateOpen {
		t.Error("Expected circuit to be OPEN before reset")
	}
	
	metrics := cb.GetMetrics()
	if metrics.FailureCount == 0 {
		t.Error("Expected non-zero failure count before reset")
	}
	
	// AIDEV-NOTE: Test reset functionality
	cb.Reset()
	
	if cb.GetState() != StateClosed {
		t.Errorf("Expected CLOSED after reset, got %s", cb.GetState())
	}
	
	resetMetrics := cb.GetMetrics()
	if resetMetrics.FailureCount != 0 {
		t.Errorf("Expected zero failure count after reset, got %d", resetMetrics.FailureCount)
	}
	
	if resetMetrics.SuccessCount != 0 {
		t.Errorf("Expected zero success count after reset, got %d", resetMetrics.SuccessCount)
	}
	
	if resetMetrics.ConsecutiveFailures != 0 {
		t.Errorf("Expected zero consecutive failures after reset, got %d", resetMetrics.ConsecutiveFailures)
	}
}

func BenchmarkCircuitBreaker_Execute(b *testing.B) {
	config := DefaultConfig()
	cb := NewCircuitBreaker(config)
	ctx := context.Background()
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			cb.Execute(ctx, func() (interface{}, error) {
				return "benchmark", nil
			})
		}
	})
}

func BenchmarkCircuitBreaker_GetMetrics(b *testing.B) {
	config := DefaultConfig()
	cb := NewCircuitBreaker(config)
	
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			cb.GetMetrics()
		}
	})
}

// AIDEV-NOTE: 250102161030 Helper function for testing timeout scenarios
func TestCircuitBreaker_ContextTimeout(t *testing.T) {
	config := DefaultConfig()
	cb := NewCircuitBreaker(config)
	
	// AIDEV-NOTE: Test with cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	
	_, err := cb.Execute(ctx, func() (interface{}, error) {
		time.Sleep(100 * time.Millisecond)
		return "success", nil
	})
	
	if err == nil {
		t.Error("Expected error with cancelled context")
	}
}

func TestCircuitBreaker_String(t *testing.T) {
	config := DefaultConfig()
	cb := NewCircuitBreaker(config)
	ctx := context.Background()
	
	// Generate some metrics
	cb.Execute(ctx, func() (interface{}, error) { return "success", nil })
	cb.Execute(ctx, func() (interface{}, error) { return nil, errors.New("failure") })
	
	str := cb.String()
	if str == "" {
		t.Error("Expected non-empty string representation")
	}
	
	// Should contain key information
	if !contains(str, "CircuitBreaker") || !contains(str, "state=") {
		t.Errorf("String representation missing key information: %s", str)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		(s == substr || (len(s) > len(substr) && 
			(s[:len(substr)] == substr || s[len(s)-len(substr):] == substr ||
				func() bool {
					for i := 1; i < len(s)-len(substr)+1; i++ {
						if s[i:i+len(substr)] == substr {
							return true
						}
					}
					return false
				}())))
}