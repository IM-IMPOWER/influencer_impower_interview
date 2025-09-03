// AIDEV-NOTE: 250102161030 Circuit breaker implementation for FastAPI client resilience
// Provides configurable failure detection and automatic recovery with exponential backoff
package circuit

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// State represents the circuit breaker state
type State int32

const (
	// StateClosed allows requests through normally
	StateClosed State = iota
	// StateOpen blocks all requests until recovery timeout
	StateOpen
	// StateHalfOpen allows limited requests to test service recovery
	StateHalfOpen
)

func (s State) String() string {
	switch s {
	case StateClosed:
		return "CLOSED"
	case StateOpen:
		return "OPEN"
	case StateHalfOpen:
		return "HALF_OPEN"
	default:
		return "UNKNOWN"
	}
}

// AIDEV-NOTE: 250102161030 Circuit breaker configuration
type Config struct {
	MaxFailures      int           `json:"max_failures"`       // Maximum failures before opening
	FailureThreshold float64       `json:"failure_threshold"`  // Failure rate threshold (0.0-1.0)
	RecoveryTimeout  time.Duration `json:"recovery_timeout"`   // Time to wait before half-open
	MonitorWindow    time.Duration `json:"monitor_window"`     // Window for failure rate calculation
	BaseBackoff      time.Duration `json:"base_backoff"`       // Base exponential backoff duration
	MaxBackoff       time.Duration `json:"max_backoff"`        // Maximum backoff duration
	HalfOpenRequests int           `json:"half_open_requests"` // Number of test requests in half-open
}

// DefaultConfig returns sensible defaults for the circuit breaker
func DefaultConfig() Config {
	return Config{
		MaxFailures:      5,
		FailureThreshold: 0.6,
		RecoveryTimeout:  30 * time.Second,
		MonitorWindow:    60 * time.Second,
		BaseBackoff:      1 * time.Second,
		MaxBackoff:       30 * time.Second,
		HalfOpenRequests: 3,
	}
}

// AIDEV-NOTE: 250102161030 Circuit breaker metrics for monitoring
type Metrics struct {
	State             State   `json:"state"`
	SuccessCount      int64   `json:"success_count"`
	FailureCount      int64   `json:"failure_count"`
	ConsecutiveFailures int64 `json:"consecutive_failures"`
	LastFailureTime   time.Time `json:"last_failure_time"`
	NextRetryTime     time.Time `json:"next_retry_time"`
	SuccessRate       float64 `json:"success_rate"`
	TotalRequests     int64   `json:"total_requests"`
	CurrentBackoff    time.Duration `json:"current_backoff"`
}

// CircuitBreaker implements the circuit breaker pattern
type CircuitBreaker struct {
	config             Config
	state              int32  // atomic State
	consecutiveFailures int64  // atomic counter
	successCount       int64  // atomic counter  
	failureCount       int64  // atomic counter
	lastFailureTime    int64  // atomic unix nano timestamp
	nextRetryTime      int64  // atomic unix nano timestamp
	halfOpenRequests   int64  // atomic counter for half-open state
	currentBackoff     int64  // atomic duration in nanoseconds
	
	mu              sync.RWMutex
	onStateChange   func(from, to State)
	fallbackFunc    func(error) (interface{}, error)
	requestLog      []requestEntry
	windowStart     time.Time
}

// requestEntry tracks individual requests for failure rate calculation
type requestEntry struct {
	timestamp time.Time
	success   bool
}

// AIDEV-NOTE: 250102161030 Create new circuit breaker with configuration
func NewCircuitBreaker(config Config) *CircuitBreaker {
	if config.MaxFailures <= 0 {
		config = DefaultConfig()
	}
	
	cb := &CircuitBreaker{
		config:      config,
		state:       int32(StateClosed),
		windowStart: time.Now(),
		requestLog:  make([]requestEntry, 0, config.MaxFailures*2),
	}
	
	atomic.StoreInt64(&cb.currentBackoff, int64(config.BaseBackoff))
	
	return cb
}

// Execute executes a function with circuit breaker protection
func (cb *CircuitBreaker) Execute(ctx context.Context, fn func() (interface{}, error)) (interface{}, error) {
	if !cb.canExecute() {
		cb.recordFailure()
		err := fmt.Errorf("circuit breaker is %s", cb.GetState())
		
		if cb.fallbackFunc != nil {
			return cb.fallbackFunc(err)
		}
		
		return nil, err
	}
	
	result, err := fn()
	
	if err != nil {
		cb.recordFailure()
		return result, err
	}
	
	cb.recordSuccess()
	return result, nil
}

// canExecute determines if a request can be executed
func (cb *CircuitBreaker) canExecute() bool {
	state := State(atomic.LoadInt32(&cb.state))
	now := time.Now()
	
	switch state {
	case StateClosed:
		return true
		
	case StateOpen:
		nextRetry := time.Unix(0, atomic.LoadInt64(&cb.nextRetryTime))
		if now.After(nextRetry) {
			// Attempt to transition to half-open
			if atomic.CompareAndSwapInt32(&cb.state, int32(StateOpen), int32(StateHalfOpen)) {
				atomic.StoreInt64(&cb.halfOpenRequests, 0)
				cb.onStateChanged(StateOpen, StateHalfOpen)
			}
			return true
		}
		return false
		
	case StateHalfOpen:
		halfOpenCount := atomic.LoadInt64(&cb.halfOpenRequests)
		if halfOpenCount < int64(cb.config.HalfOpenRequests) {
			atomic.AddInt64(&cb.halfOpenRequests, 1)
			return true
		}
		return false
		
	default:
		return false
	}
}

// recordSuccess records a successful request
func (cb *CircuitBreaker) recordSuccess() {
	atomic.AddInt64(&cb.successCount, 1)
	atomic.StoreInt64(&cb.consecutiveFailures, 0)
	
	// Reset backoff on success
	atomic.StoreInt64(&cb.currentBackoff, int64(cb.config.BaseBackoff))
	
	state := State(atomic.LoadInt32(&cb.state))
	
	cb.mu.Lock()
	cb.addRequestEntry(true)
	cb.mu.Unlock()
	
	// Transition from half-open to closed if we have enough successful requests
	if state == StateHalfOpen {
		halfOpenCount := atomic.LoadInt64(&cb.halfOpenRequests)
		if halfOpenCount >= int64(cb.config.HalfOpenRequests) {
			if atomic.CompareAndSwapInt32(&cb.state, int32(StateHalfOpen), int32(StateClosed)) {
				cb.onStateChanged(StateHalfOpen, StateClosed)
			}
		}
	}
}

// recordFailure records a failed request  
func (cb *CircuitBreaker) recordFailure() {
	atomic.AddInt64(&cb.failureCount, 1)
	consecutiveFailures := atomic.AddInt64(&cb.consecutiveFailures, 1)
	atomic.StoreInt64(&cb.lastFailureTime, time.Now().UnixNano())
	
	cb.mu.Lock()
	cb.addRequestEntry(false)
	failureRate := cb.calculateFailureRate()
	cb.mu.Unlock()
	
	state := State(atomic.LoadInt32(&cb.state))
	
	// Check if we should open the circuit
	shouldOpen := (consecutiveFailures >= int64(cb.config.MaxFailures)) ||
		(failureRate >= cb.config.FailureThreshold)
	
	if (state == StateClosed || state == StateHalfOpen) && shouldOpen {
		// Calculate next retry time with exponential backoff
		currentBackoff := time.Duration(atomic.LoadInt64(&cb.currentBackoff))
		nextBackoff := time.Duration(float64(currentBackoff) * 2.0)
		if nextBackoff > cb.config.MaxBackoff {
			nextBackoff = cb.config.MaxBackoff
		}
		atomic.StoreInt64(&cb.currentBackoff, int64(nextBackoff))
		
		nextRetryTime := time.Now().Add(currentBackoff)
		atomic.StoreInt64(&cb.nextRetryTime, nextRetryTime.UnixNano())
		
		if atomic.CompareAndSwapInt32(&cb.state, int32(state), int32(StateOpen)) {
			cb.onStateChanged(state, StateOpen)
		}
	}
}

// addRequestEntry adds a request to the sliding window
func (cb *CircuitBreaker) addRequestEntry(success bool) {
	now := time.Now()
	
	// Remove old entries outside the monitoring window
	cutoff := now.Add(-cb.config.MonitorWindow)
	newLog := make([]requestEntry, 0, len(cb.requestLog))
	
	for _, entry := range cb.requestLog {
		if entry.timestamp.After(cutoff) {
			newLog = append(newLog, entry)
		}
	}
	
	// Add new entry
	newLog = append(newLog, requestEntry{
		timestamp: now,
		success:   success,
	})
	
	cb.requestLog = newLog
	
	// Update window start if this is the first entry in a new window
	if cb.windowStart.Before(cutoff) {
		cb.windowStart = now
	}
}

// calculateFailureRate calculates the current failure rate
func (cb *CircuitBreaker) calculateFailureRate() float64 {
	if len(cb.requestLog) == 0 {
		return 0.0
	}
	
	var failures, total int
	for _, entry := range cb.requestLog {
		total++
		if !entry.success {
			failures++
		}
	}
	
	if total == 0 {
		return 0.0
	}
	
	return float64(failures) / float64(total)
}

// GetState returns the current circuit breaker state
func (cb *CircuitBreaker) GetState() State {
	return State(atomic.LoadInt32(&cb.state))
}

// GetMetrics returns current circuit breaker metrics
func (cb *CircuitBreaker) GetMetrics() Metrics {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	
	successCount := atomic.LoadInt64(&cb.successCount)
	failureCount := atomic.LoadInt64(&cb.failureCount)
	totalRequests := successCount + failureCount
	
	var successRate float64
	if totalRequests > 0 {
		successRate = float64(successCount) / float64(totalRequests)
	}
	
	return Metrics{
		State:               cb.GetState(),
		SuccessCount:        successCount,
		FailureCount:        failureCount,
		ConsecutiveFailures: atomic.LoadInt64(&cb.consecutiveFailures),
		LastFailureTime:     time.Unix(0, atomic.LoadInt64(&cb.lastFailureTime)),
		NextRetryTime:       time.Unix(0, atomic.LoadInt64(&cb.nextRetryTime)),
		SuccessRate:         successRate,
		TotalRequests:       totalRequests,
		CurrentBackoff:      time.Duration(atomic.LoadInt64(&cb.currentBackoff)),
	}
}

// IsHealthy returns whether the circuit breaker is healthy (closed state)
func (cb *CircuitBreaker) IsHealthy() bool {
	return cb.GetState() == StateClosed
}

// Reset resets the circuit breaker to closed state  
func (cb *CircuitBreaker) Reset() {
	atomic.StoreInt32(&cb.state, int32(StateClosed))
	atomic.StoreInt64(&cb.consecutiveFailures, 0)
	atomic.StoreInt64(&cb.successCount, 0)
	atomic.StoreInt64(&cb.failureCount, 0)
	atomic.StoreInt64(&cb.halfOpenRequests, 0)
	atomic.StoreInt64(&cb.currentBackoff, int64(cb.config.BaseBackoff))
	
	cb.mu.Lock()
	cb.requestLog = cb.requestLog[:0]
	cb.windowStart = time.Now()
	cb.mu.Unlock()
}

// SetFallback sets a fallback function to call when circuit is open
func (cb *CircuitBreaker) SetFallback(fn func(error) (interface{}, error)) {
	cb.mu.Lock()
	cb.fallbackFunc = fn
	cb.mu.Unlock()
}

// SetOnStateChange sets a callback for state changes
func (cb *CircuitBreaker) SetOnStateChange(fn func(from, to State)) {
	cb.mu.Lock()
	cb.onStateChange = fn
	cb.mu.Unlock()
}

// onStateChanged calls the state change callback if set
func (cb *CircuitBreaker) onStateChanged(from, to State) {
	if cb.onStateChange != nil {
		go cb.onStateChange(from, to)
	}
}

// AIDEV-NOTE: 250102161030 Utility functions for monitoring and debugging

// GetConfig returns the current circuit breaker configuration
func (cb *CircuitBreaker) GetConfig() Config {
	return cb.config
}

// String returns a string representation of the circuit breaker
func (cb *CircuitBreaker) String() string {
	metrics := cb.GetMetrics()
	return fmt.Sprintf("CircuitBreaker{state=%s, success=%d, failures=%d, rate=%.2f%%}",
		metrics.State, metrics.SuccessCount, metrics.FailureCount, metrics.SuccessRate*100)
}