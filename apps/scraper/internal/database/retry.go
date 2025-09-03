// AIDEV-NOTE: 250903170400 Database retry logic with exponential backoff and circuit breaker
// Provides resilient database operations with intelligent retry mechanisms
package database

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/jmoiron/sqlx"
	"kol-scraper/internal/errors"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: 250903170401 Retry configuration for database operations
type RetryConfig struct {
	MaxAttempts     int           `json:"max_attempts"`
	InitialDelay    time.Duration `json:"initial_delay"`
	MaxDelay        time.Duration `json:"max_delay"`
	BackoffFactor   float64       `json:"backoff_factor"`
	JitterFactor    float64       `json:"jitter_factor"`
	RetriableErrors []string      `json:"retriable_errors"`
	EnableJitter    bool          `json:"enable_jitter"`
}

// AIDEV-NOTE: 250903170402 Resilient database client with retry capabilities
type ResilientDB struct {
	db          *sqlx.DB
	poolManager *PoolManager
	logger      *logger.Logger
	config      *RetryConfig
	metrics     *RetryMetrics
	mu          sync.RWMutex
}

// AIDEV-NOTE: 250903170403 Retry operation metrics
type RetryMetrics struct {
	mu                sync.RWMutex
	TotalOperations   int64                    `json:"total_operations"`
	RetriedOperations int64                    `json:"retried_operations"`
	FailedOperations  int64                    `json:"failed_operations"`
	RetrySuccess      int64                    `json:"retry_success"`
	RetryAttempts     map[string]int64         `json:"retry_attempts"`
	AvgRetryDelay     time.Duration            `json:"avg_retry_delay"`
	ErrorDistribution map[string]int64         `json:"error_distribution"`
	CircuitBreakerTrips int64                  `json:"circuit_breaker_trips"`
}

// AIDEV-NOTE: 250903170404 Operation context for retry tracking
type OperationContext struct {
	OperationType string
	Query         string
	Args          []interface{}
	StartTime     time.Time
	AttemptCount  int
	LastError     error
	Context       context.Context
}

// NewResilientDB creates a new resilient database client with retry capabilities
func NewResilientDB(db *sqlx.DB, poolManager *PoolManager, logger *logger.Logger) *ResilientDB {
	config := &RetryConfig{
		MaxAttempts:   5,
		InitialDelay:  100 * time.Millisecond,
		MaxDelay:      30 * time.Second,
		BackoffFactor: 2.0,
		JitterFactor:  0.1,
		EnableJitter:  true,
		RetriableErrors: []string{
			"connection refused",
			"connection reset",
			"connection timeout",
			"temporary failure",
			"deadlock detected",
			"lock timeout",
			"connection lost",
			"server closed the connection",
			"network error",
			"i/o timeout",
		},
	}

	return &ResilientDB{
		db:          db,
		poolManager: poolManager,
		logger:      logger,
		config:      config,
		metrics: &RetryMetrics{
			RetryAttempts:     make(map[string]int64),
			ErrorDistribution: make(map[string]int64),
		},
	}
}

// AIDEV-NOTE: 250903170405 Resilient query execution with retry logic
func (rdb *ResilientDB) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	operation := &OperationContext{
		OperationType: "query",
		Query:         query,
		Args:          args,
		StartTime:     time.Now(),
		Context:       ctx,
	}

	return rdb.executeWithRetry(operation, func(ctx context.Context) (interface{}, error) {
		// AIDEV-NOTE: 250903170406 Check circuit breaker before execution
		if !rdb.poolManager.IsHealthy() {
			rdb.recordCircuitBreakerTrip()
			return nil, errors.New(errors.CategoryDatabase, errors.CodeCircuitBreakerOpen,
				"Database circuit breaker is open")
		}

		rows, err := rdb.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, rdb.wrapDatabaseError(err, "query", query)
		}
		return rows, nil
	})
}

// ExecContext executes a query with retry logic
func (rdb *ResilientDB) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	operation := &OperationContext{
		OperationType: "exec",
		Query:         query,
		Args:          args,
		StartTime:     time.Now(),
		Context:       ctx,
	}

	result, err := rdb.executeWithRetry(operation, func(ctx context.Context) (interface{}, error) {
		if !rdb.poolManager.IsHealthy() {
			rdb.recordCircuitBreakerTrip()
			return nil, errors.New(errors.CategoryDatabase, errors.CodeCircuitBreakerOpen,
				"Database circuit breaker is open")
		}

		result, err := rdb.db.ExecContext(ctx, query, args...)
		if err != nil {
			return nil, rdb.wrapDatabaseError(err, "exec", query)
		}
		return result, nil
	})

	if err != nil {
		return nil, err
	}

	return result.(sql.Result), nil
}

// GetContext performs a single row query with retry logic
func (rdb *ResilientDB) GetContext(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	operation := &OperationContext{
		OperationType: "get",
		Query:         query,
		Args:          args,
		StartTime:     time.Now(),
		Context:       ctx,
	}

	_, err := rdb.executeWithRetry(operation, func(ctx context.Context) (interface{}, error) {
		if !rdb.poolManager.IsHealthy() {
			rdb.recordCircuitBreakerTrip()
			return nil, errors.New(errors.CategoryDatabase, errors.CodeCircuitBreakerOpen,
				"Database circuit breaker is open")
		}

		err := rdb.db.GetContext(ctx, dest, query, args...)
		if err != nil {
			return nil, rdb.wrapDatabaseError(err, "get", query)
		}
		return dest, nil
	})

	return err
}

// SelectContext performs a multi-row query with retry logic
func (rdb *ResilientDB) SelectContext(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	operation := &OperationContext{
		OperationType: "select",
		Query:         query,
		Args:          args,
		StartTime:     time.Now(),
		Context:       ctx,
	}

	_, err := rdb.executeWithRetry(operation, func(ctx context.Context) (interface{}, error) {
		if !rdb.poolManager.IsHealthy() {
			rdb.recordCircuitBreakerTrip()
			return nil, errors.New(errors.CategoryDatabase, errors.CodeCircuitBreakerOpen,
				"Database circuit breaker is open")
		}

		err := rdb.db.SelectContext(ctx, dest, query, args...)
		if err != nil {
			return nil, rdb.wrapDatabaseError(err, "select", query)
		}
		return dest, nil
	})

	return err
}

// AIDEV-NOTE: 250903170407 Transaction with retry logic
func (rdb *ResilientDB) TransactionWithRetry(ctx context.Context, fn func(*sqlx.Tx) error) error {
	operation := &OperationContext{
		OperationType: "transaction",
		StartTime:     time.Now(),
		Context:       ctx,
	}

	_, err := rdb.executeWithRetry(operation, func(ctx context.Context) (interface{}, error) {
		if !rdb.poolManager.IsHealthy() {
			rdb.recordCircuitBreakerTrip()
			return nil, errors.New(errors.CategoryDatabase, errors.CodeCircuitBreakerOpen,
				"Database circuit breaker is open")
		}

		tx, err := rdb.db.BeginTxx(ctx, nil)
		if err != nil {
			return nil, rdb.wrapDatabaseError(err, "begin_transaction", "")
		}

		defer func() {
			if p := recover(); p != nil {
				_ = tx.Rollback()
				panic(p)
			}
		}()

		if err := fn(tx); err != nil {
			_ = tx.Rollback()
			return nil, rdb.wrapDatabaseError(err, "transaction_operation", "")
		}

		if err := tx.Commit(); err != nil {
			return nil, rdb.wrapDatabaseError(err, "commit_transaction", "")
		}

		return nil, nil
	})

	return err
}

// AIDEV-NOTE: 250903170408 Core retry execution logic with exponential backoff
func (rdb *ResilientDB) executeWithRetry(operation *OperationContext, fn func(context.Context) (interface{}, error)) (interface{}, error) {
	rdb.recordOperation()

	var result interface{}
	var err error

	for attempt := 1; attempt <= rdb.config.MaxAttempts; attempt++ {
		operation.AttemptCount = attempt

		// AIDEV-NOTE: 250903170409 Check context cancellation
		if operation.Context.Err() != nil {
			return nil, operation.Context.Err()
		}

		// AIDEV-NOTE: 250903170410 Execute the operation
		result, err = fn(operation.Context)
		if err == nil {
			// Success - record metrics and return
			if attempt > 1 {
				rdb.recordRetrySuccess(operation)
			}
			return result, nil
		}

		operation.LastError = err

		// AIDEV-NOTE: 250903170411 Check if error is retriable
		if !rdb.isRetriableError(err) || attempt == rdb.config.MaxAttempts {
			rdb.recordFailedOperation(operation)
			return nil, err
		}

		// AIDEV-NOTE: 250903170412 Calculate delay with exponential backoff
		delay := rdb.calculateDelay(attempt)
		rdb.recordRetryAttempt(operation, delay)

		// AIDEV-NOTE: 250903170413 Wait before retrying
		select {
		case <-time.After(delay):
			continue
		case <-operation.Context.Done():
			return nil, operation.Context.Err()
		}
	}

	rdb.recordFailedOperation(operation)
	return nil, err
}

// calculateDelay computes the retry delay using exponential backoff with jitter
func (rdb *ResilientDB) calculateDelay(attempt int) time.Duration {
	// AIDEV-NOTE: 250903170414 Exponential backoff calculation
	delay := float64(rdb.config.InitialDelay) * math.Pow(rdb.config.BackoffFactor, float64(attempt-1))

	// AIDEV-NOTE: 250903170415 Apply maximum delay cap
	if delay > float64(rdb.config.MaxDelay) {
		delay = float64(rdb.config.MaxDelay)
	}

	// AIDEV-NOTE: 250903170416 Add jitter to prevent thundering herd
	if rdb.config.EnableJitter {
		jitter := delay * rdb.config.JitterFactor * (rand.Float64()*2 - 1)
		delay += jitter
	}

	// AIDEV-NOTE: 250903170417 Ensure minimum positive delay
	if delay < float64(rdb.config.InitialDelay) {
		delay = float64(rdb.config.InitialDelay)
	}

	return time.Duration(delay)
}

// isRetriableError determines if an error should trigger a retry
func (rdb *ResilientDB) isRetriableError(err error) bool {
	if err == nil {
		return false
	}

	errorMsg := strings.ToLower(err.Error())

	// AIDEV-NOTE: 250903170418 Check for known retriable error patterns
	for _, pattern := range rdb.config.RetriableErrors {
		if strings.Contains(errorMsg, strings.ToLower(pattern)) {
			return true
		}
	}

	// AIDEV-NOTE: 250903170419 Check for specific database error conditions
	if strings.Contains(errorMsg, "deadlock") ||
		strings.Contains(errorMsg, "lock timeout") ||
		strings.Contains(errorMsg, "connection") ||
		strings.Contains(errorMsg, "timeout") ||
		strings.Contains(errorMsg, "temporary") {
		return true
	}

	// AIDEV-NOTE: 250903170420 Check KOLError types
	if kolErr, ok := err.(errors.KOLError); ok {
		return kolErr.IsRetryable()
	}

	return false
}

// wrapDatabaseError wraps database errors with additional context
func (rdb *ResilientDB) wrapDatabaseError(err error, operation, query string) error {
	if err == nil {
		return nil
	}

	// AIDEV-NOTE: 250903170421 Classify error types for better handling
	errorMsg := err.Error()
	var code errors.ErrorCode

	if err == sql.ErrNoRows {
		code = errors.CodeRecordNotFound
	} else if strings.Contains(errorMsg, "duplicate") {
		code = errors.CodeDuplicateRecord
	} else if strings.Contains(errorMsg, "connection") {
		code = errors.CodeDatabaseConnection
	} else {
		code = errors.CodeDatabaseQuery
	}

	return errors.WrapDatabase(err, operation).
		WithContext("query", query).
		WithContext("operation", operation)
}

// AIDEV-NOTE: 250903170422 Metrics recording methods
func (rdb *ResilientDB) recordOperation() {
	rdb.metrics.mu.Lock()
	defer rdb.metrics.mu.Unlock()
	rdb.metrics.TotalOperations++
}

func (rdb *ResilientDB) recordRetryAttempt(operation *OperationContext, delay time.Duration) {
	rdb.metrics.mu.Lock()
	defer rdb.metrics.mu.Unlock()

	rdb.metrics.RetriedOperations++
	rdb.metrics.RetryAttempts[operation.OperationType]++

	// Update average retry delay
	totalRetries := rdb.metrics.RetriedOperations
	if totalRetries > 1 {
		avgDelay := rdb.metrics.AvgRetryDelay
		rdb.metrics.AvgRetryDelay = time.Duration(
			(int64(avgDelay)*(totalRetries-1) + int64(delay)) / totalRetries,
		)
	} else {
		rdb.metrics.AvgRetryDelay = delay
	}

	rdb.logger.Debug("Database operation retry", logger.Fields{
		"operation":     operation.OperationType,
		"attempt":       operation.AttemptCount,
		"delay_ms":      delay.Milliseconds(),
		"error":         operation.LastError.Error(),
	})
}

func (rdb *ResilientDB) recordRetrySuccess(operation *OperationContext) {
	rdb.metrics.mu.Lock()
	defer rdb.metrics.mu.Unlock()

	rdb.metrics.RetrySuccess++

	rdb.logger.Info("Database operation retry succeeded", logger.Fields{
		"operation":     operation.OperationType,
		"total_attempts": operation.AttemptCount,
		"duration_ms":   time.Since(operation.StartTime).Milliseconds(),
	})
}

func (rdb *ResilientDB) recordFailedOperation(operation *OperationContext) {
	rdb.metrics.mu.Lock()
	defer rdb.metrics.mu.Unlock()

	rdb.metrics.FailedOperations++

	if operation.LastError != nil {
		errorType := rdb.classifyError(operation.LastError)
		rdb.metrics.ErrorDistribution[errorType]++
	}

	rdb.logger.ErrorLog(
		errors.WrapDatabase(operation.LastError, operation.OperationType),
		"database_operation_failed_after_retries",
		logger.Fields{
			"operation":      operation.OperationType,
			"total_attempts": operation.AttemptCount,
			"duration_ms":    time.Since(operation.StartTime).Milliseconds(),
			"query":          operation.Query,
		},
	)
}

func (rdb *ResilientDB) recordCircuitBreakerTrip() {
	rdb.metrics.mu.Lock()
	defer rdb.metrics.mu.Unlock()

	rdb.metrics.CircuitBreakerTrips++

	rdb.logger.Warn("Database circuit breaker prevented operation", logger.Fields{
		"total_trips": rdb.metrics.CircuitBreakerTrips,
	})
}

// classifyError categorizes errors for metrics
func (rdb *ResilientDB) classifyError(err error) string {
	if err == nil {
		return "unknown"
	}

	errorMsg := strings.ToLower(err.Error())

	if strings.Contains(errorMsg, "connection") {
		return "connection_error"
	}
	if strings.Contains(errorMsg, "timeout") {
		return "timeout_error"
	}
	if strings.Contains(errorMsg, "deadlock") {
		return "deadlock_error"
	}
	if strings.Contains(errorMsg, "duplicate") {
		return "constraint_error"
	}
	if err == sql.ErrNoRows {
		return "not_found_error"
	}

	return "other_error"
}

// GetRetryMetrics returns current retry metrics
func (rdb *ResilientDB) GetRetryMetrics() *RetryMetrics {
	rdb.metrics.mu.RLock()
	defer rdb.metrics.mu.RUnlock()

	// Create a deep copy
	attempts := make(map[string]int64)
	for k, v := range rdb.metrics.RetryAttempts {
		attempts[k] = v
	}

	distribution := make(map[string]int64)
	for k, v := range rdb.metrics.ErrorDistribution {
		distribution[k] = v
	}

	return &RetryMetrics{
		TotalOperations:     rdb.metrics.TotalOperations,
		RetriedOperations:   rdb.metrics.RetriedOperations,
		FailedOperations:    rdb.metrics.FailedOperations,
		RetrySuccess:        rdb.metrics.RetrySuccess,
		RetryAttempts:       attempts,
		AvgRetryDelay:       rdb.metrics.AvgRetryDelay,
		ErrorDistribution:   distribution,
		CircuitBreakerTrips: rdb.metrics.CircuitBreakerTrips,
	}
}

// UpdateRetryConfig allows dynamic retry configuration updates
func (rdb *ResilientDB) UpdateRetryConfig(config *RetryConfig) {
	rdb.mu.Lock()
	defer rdb.mu.Unlock()

	rdb.config = config

	rdb.logger.Info("Retry configuration updated", logger.Fields{
		"max_attempts":   config.MaxAttempts,
		"initial_delay":  config.InitialDelay,
		"max_delay":      config.MaxDelay,
		"backoff_factor": config.BackoffFactor,
		"jitter_enabled": config.EnableJitter,
	})
}

// AIDEV-NOTE: 250903170423 Health check with retry awareness
func (rdb *ResilientDB) HealthCheck(ctx context.Context) error {
	// Use internal retry mechanism for health check
	_, err := rdb.executeWithRetry(&OperationContext{
		OperationType: "health_check",
		StartTime:     time.Now(),
		Context:       ctx,
	}, func(ctx context.Context) (interface{}, error) {
		var result int
		err := rdb.db.GetContext(ctx, &result, "SELECT 1")
		return result, err
	})

	return err
}

// GetSuccessRate returns the operation success rate
func (rdb *ResilientDB) GetSuccessRate() float64 {
	rdb.metrics.mu.RLock()
	defer rdb.metrics.mu.RUnlock()

	if rdb.metrics.TotalOperations == 0 {
		return 1.0
	}

	successful := rdb.metrics.TotalOperations - rdb.metrics.FailedOperations
	return float64(successful) / float64(rdb.metrics.TotalOperations)
}

// GetRetryRate returns the percentage of operations that required retries
func (rdb *ResilientDB) GetRetryRate() float64 {
	rdb.metrics.mu.RLock()
	defer rdb.metrics.mu.RUnlock()

	if rdb.metrics.TotalOperations == 0 {
		return 0.0
	}

	return float64(rdb.metrics.RetriedOperations) / float64(rdb.metrics.TotalOperations)
}

// Close gracefully closes the resilient database client
func (rdb *ResilientDB) Close() error {
	rdb.logger.Info("Closing resilient database client", logger.Fields{
		"total_operations":   rdb.metrics.TotalOperations,
		"retried_operations": rdb.metrics.RetriedOperations,
		"success_rate":       rdb.GetSuccessRate(),
		"retry_rate":         rdb.GetRetryRate(),
	})

	return nil
}