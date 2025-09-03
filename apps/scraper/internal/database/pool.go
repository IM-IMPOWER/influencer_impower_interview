// AIDEV-NOTE: 250903170000 Advanced database connection pool management
// Optimizes connection pooling, monitoring, and automatic recovery for production workloads
package database

import (
	"context"
	"database/sql"
	"fmt"
	"sync"
	"time"

	"github.com/jmoiron/sqlx"
	"kol-scraper/internal/errors"
	"kol-scraper/pkg/config"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: 250903170001 Pool configuration for production optimization
type PoolConfig struct {
	MaxOpenConns        int           `json:"max_open_conns"`
	MaxIdleConns        int           `json:"max_idle_conns"`
	ConnMaxLifetime     time.Duration `json:"conn_max_lifetime"`
	ConnMaxIdleTime     time.Duration `json:"conn_max_idle_time"`
	HealthCheckInterval time.Duration `json:"health_check_interval"`
	SlowQueryThreshold  time.Duration `json:"slow_query_threshold"`
	EnableMonitoring    bool          `json:"enable_monitoring"`
	EnableSlowQueryLog  bool          `json:"enable_slow_query_log"`
}

// AIDEV-NOTE: 250903170002 Pool manager with health monitoring and metrics
type PoolManager struct {
	db          *sqlx.DB
	config      *PoolConfig
	logger      *logger.Logger
	metrics     *PoolMetrics
	healthCheck *HealthCheck
	mu          sync.RWMutex
	stopCh      chan struct{}
	wg          sync.WaitGroup
}

// AIDEV-NOTE: 250903170003 Pool metrics collection
type PoolMetrics struct {
	mu                    sync.RWMutex
	ConnectionsOpened     int64                    `json:"connections_opened"`
	ConnectionsClosed     int64                    `json:"connections_closed"`
	ConnectionsInUse      int32                    `json:"connections_in_use"`
	ConnectionsIdle       int32                    `json:"connections_idle"`
	SlowQueries          int64                    `json:"slow_queries"`
	FailedConnections    int64                    `json:"failed_connections"`
	QueryLatency         map[string]time.Duration `json:"query_latency"`
	LastHealthCheck      time.Time                `json:"last_health_check"`
	HealthCheckDuration  time.Duration            `json:"health_check_duration"`
	CircuitBreakerTrips  int64                    `json:"circuit_breaker_trips"`
}

// AIDEV-NOTE: 250903170004 Health check configuration
type HealthCheck struct {
	Enabled         bool          `json:"enabled"`
	Interval        time.Duration `json:"interval"`
	Timeout         time.Duration `json:"timeout"`
	MaxFailures     int           `json:"max_failures"`
	FailureCount    int           `json:"failure_count"`
	LastCheck       time.Time     `json:"last_check"`
	Status          string        `json:"status"`
	CircuitBreaker  *CircuitBreaker
}

// AIDEV-NOTE: 250903170005 Circuit breaker for database connections
type CircuitBreaker struct {
	mu              sync.RWMutex
	maxRequests     int
	interval        time.Duration
	timeout         time.Duration
	consecutiveFailures int
	state           CircuitState
	lastFailure     time.Time
	nextRetry       time.Time
}

type CircuitState string

const (
	StateClosed     CircuitState = "closed"
	StateOpen       CircuitState = "open"
	StateHalfOpen   CircuitState = "half_open"
)

// NewPoolManager creates a new optimized database pool manager
func NewPoolManager(db *sqlx.DB, cfg *config.Config, log *logger.Logger) *PoolManager {
	// AIDEV-NOTE: 250903170006 Calculate optimal pool settings based on system resources
	poolConfig := calculateOptimalPoolSettings(cfg)

	pm := &PoolManager{
		db:     db,
		config: poolConfig,
		logger: log,
		metrics: &PoolMetrics{
			QueryLatency: make(map[string]time.Duration),
		},
		healthCheck: &HealthCheck{
			Enabled:     true,
			Interval:    30 * time.Second,
			Timeout:     5 * time.Second,
			MaxFailures: 3,
			Status:      "unknown",
			CircuitBreaker: &CircuitBreaker{
				maxRequests: 10,
				interval:    60 * time.Second,
				timeout:     30 * time.Second,
				state:       StateClosed,
			},
		},
		stopCh: make(chan struct{}),
	}

	// AIDEV-NOTE: 250903170007 Apply connection pool optimizations
	pm.applyPoolOptimizations()

	// AIDEV-NOTE: 250903170008 Start monitoring goroutines
	if poolConfig.EnableMonitoring {
		pm.startMonitoring()
	}

	return pm
}

// applyPoolOptimizations configures the database connection pool for optimal performance
func (pm *PoolManager) applyPoolOptimizations() {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	// AIDEV-NOTE: 250903170009 Configure connection pool parameters
	pm.db.SetMaxOpenConns(pm.config.MaxOpenConns)
	pm.db.SetMaxIdleConns(pm.config.MaxIdleConns)
	pm.db.SetConnMaxLifetime(pm.config.ConnMaxLifetime)
	pm.db.SetConnMaxIdleTime(pm.config.ConnMaxIdleTime)

	pm.logger.Info("Database pool optimizations applied", logger.Fields{
		"max_open_conns":     pm.config.MaxOpenConns,
		"max_idle_conns":     pm.config.MaxIdleConns,
		"conn_max_lifetime":  pm.config.ConnMaxLifetime,
		"conn_max_idle_time": pm.config.ConnMaxIdleTime,
	})
}

// calculateOptimalPoolSettings determines optimal pool settings based on configuration
func calculateOptimalPoolSettings(cfg *config.Config) *PoolConfig {
	// AIDEV-NOTE: 250903170010 Calculate optimal settings based on expected load
	maxOpen := cfg.MaxConnections
	if maxOpen <= 0 {
		maxOpen = 25 // Production default
	}

	maxIdle := cfg.MaxIdleConns
	if maxIdle <= 0 {
		maxIdle = maxOpen / 4 // 25% of max connections
		if maxIdle < 5 {
			maxIdle = 5 // Minimum idle connections
		}
	}

	return &PoolConfig{
		MaxOpenConns:        maxOpen,
		MaxIdleConns:        maxIdle,
		ConnMaxLifetime:     time.Duration(cfg.ConnMaxLifetime) * time.Second,
		ConnMaxIdleTime:     15 * time.Minute, // Close idle connections after 15 minutes
		HealthCheckInterval: 30 * time.Second,
		SlowQueryThreshold:  1 * time.Second,
		EnableMonitoring:    true,
		EnableSlowQueryLog:  cfg.Environment == "production",
	}
}

// startMonitoring starts background monitoring goroutines
func (pm *PoolManager) startMonitoring() {
	pm.wg.Add(2)

	// AIDEV-NOTE: 250903170011 Start health check monitoring
	go pm.runHealthChecks()

	// AIDEV-NOTE: 250903170012 Start metrics collection
	go pm.collectMetrics()
}

// runHealthChecks performs periodic database health checks
func (pm *PoolManager) runHealthChecks() {
	defer pm.wg.Done()

	ticker := time.NewTicker(pm.healthCheck.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			pm.performHealthCheck()
		case <-pm.stopCh:
			return
		}
	}
}

// performHealthCheck executes a database health check
func (pm *PoolManager) performHealthCheck() {
	start := time.Now()
	ctx, cancel := context.WithTimeout(context.Background(), pm.healthCheck.Timeout)
	defer cancel()

	pm.mu.Lock()
	defer pm.mu.Unlock()

	// AIDEV-NOTE: 250903170013 Execute health check query
	var result int
	err := pm.db.GetContext(ctx, &result, "SELECT 1")
	duration := time.Since(start)

	pm.metrics.LastHealthCheck = start
	pm.metrics.HealthCheckDuration = duration

	if err != nil {
		pm.healthCheck.FailureCount++
		pm.healthCheck.Status = "unhealthy"
		pm.metrics.FailedConnections++

		// AIDEV-NOTE: 250903170014 Trigger circuit breaker if failures exceed threshold
		if pm.healthCheck.FailureCount >= pm.healthCheck.MaxFailures {
			pm.healthCheck.CircuitBreaker.recordFailure()
			pm.metrics.CircuitBreakerTrips++
		}

		pm.logger.ErrorLog(errors.WrapDatabase(err, "health_check"), "database_health_check_failed", logger.Fields{
			"failure_count":      pm.healthCheck.FailureCount,
			"max_failures":       pm.healthCheck.MaxFailures,
			"duration_ms":        duration.Milliseconds(),
			"circuit_breaker_state": pm.healthCheck.CircuitBreaker.state,
		})
	} else {
		// AIDEV-NOTE: 250903170015 Reset failure count on successful health check
		pm.healthCheck.FailureCount = 0
		pm.healthCheck.Status = "healthy"
		pm.healthCheck.CircuitBreaker.recordSuccess()

		pm.logger.Debug("Database health check succeeded", logger.Fields{
			"duration_ms": duration.Milliseconds(),
		})
	}

	pm.healthCheck.LastCheck = time.Now()
}

// collectMetrics periodically collects database pool metrics
func (pm *PoolManager) collectMetrics() {
	defer pm.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			pm.updatePoolMetrics()
		case <-pm.stopCh:
			return
		}
	}
}

// updatePoolMetrics updates internal pool metrics
func (pm *PoolManager) updatePoolMetrics() {
	stats := pm.db.Stats()

	pm.metrics.mu.Lock()
	defer pm.metrics.mu.Unlock()

	pm.metrics.ConnectionsInUse = int32(stats.InUse)
	pm.metrics.ConnectionsIdle = int32(stats.Idle)
	pm.metrics.ConnectionsOpened = stats.OpenConnections
	pm.metrics.ConnectionsClosed = stats.MaxLifetimeClosed + stats.MaxIdleClosed

	// AIDEV-NOTE: 250903170016 Log metrics if monitoring is enabled
	if pm.config.EnableMonitoring {
		pm.logger.Debug("Database pool metrics", logger.Fields{
			"open_connections":      stats.OpenConnections,
			"in_use":               stats.InUse,
			"idle":                 stats.Idle,
			"wait_count":           stats.WaitCount,
			"wait_duration_ms":     stats.WaitDuration.Milliseconds(),
			"max_idle_closed":      stats.MaxIdleClosed,
			"max_lifetime_closed":  stats.MaxLifetimeClosed,
		})
	}
}

// GetMetrics returns current pool metrics
func (pm *PoolManager) GetMetrics() *PoolMetrics {
	pm.metrics.mu.RLock()
	defer pm.metrics.mu.RUnlock()

	// Create a copy to avoid race conditions
	return &PoolMetrics{
		ConnectionsOpened:    pm.metrics.ConnectionsOpened,
		ConnectionsClosed:    pm.metrics.ConnectionsClosed,
		ConnectionsInUse:     pm.metrics.ConnectionsInUse,
		ConnectionsIdle:      pm.metrics.ConnectionsIdle,
		SlowQueries:         pm.metrics.SlowQueries,
		FailedConnections:   pm.metrics.FailedConnections,
		LastHealthCheck:     pm.metrics.LastHealthCheck,
		HealthCheckDuration: pm.metrics.HealthCheckDuration,
		CircuitBreakerTrips: pm.metrics.CircuitBreakerTrips,
	}
}

// GetHealthStatus returns the current health status
func (pm *PoolManager) GetHealthStatus() map[string]interface{} {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	return map[string]interface{}{
		"status":               pm.healthCheck.Status,
		"last_check":           pm.healthCheck.LastCheck,
		"failure_count":        pm.healthCheck.FailureCount,
		"max_failures":         pm.healthCheck.MaxFailures,
		"circuit_breaker_state": pm.healthCheck.CircuitBreaker.state,
		"next_retry":           pm.healthCheck.CircuitBreaker.nextRetry,
	}
}

// IsHealthy returns true if the database is healthy
func (pm *PoolManager) IsHealthy() bool {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	return pm.healthCheck.Status == "healthy" && 
		   pm.healthCheck.CircuitBreaker.state != StateOpen
}

// recordFailure records a circuit breaker failure
func (cb *CircuitBreaker) recordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.consecutiveFailures++
	cb.lastFailure = time.Now()

	// AIDEV-NOTE: 250903170017 Open circuit if failures exceed threshold
	if cb.consecutiveFailures >= cb.maxRequests {
		cb.state = StateOpen
		cb.nextRetry = time.Now().Add(cb.timeout)
	}
}

// recordSuccess records a circuit breaker success
func (cb *CircuitBreaker) recordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.consecutiveFailures = 0

	if cb.state == StateHalfOpen {
		cb.state = StateClosed
	}
}

// canExecute checks if the circuit breaker allows execution
func (cb *CircuitBreaker) canExecute() bool {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	switch cb.state {
	case StateClosed:
		return true
	case StateOpen:
		if time.Now().After(cb.nextRetry) {
			cb.mu.RUnlock()
			cb.mu.Lock()
			cb.state = StateHalfOpen
			cb.mu.Unlock()
			cb.mu.RLock()
			return true
		}
		return false
	case StateHalfOpen:
		return true
	default:
		return false
	}
}

// Close gracefully shuts down the pool manager
func (pm *PoolManager) Close() error {
	close(pm.stopCh)
	pm.wg.Wait()

	pm.logger.Info("Database pool manager shutdown completed")
	return nil
}

// AIDEV-NOTE: 250903170018 Query execution with monitoring and slow query detection
func (pm *PoolManager) ExecContextWithMonitoring(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	if !pm.healthCheck.CircuitBreaker.canExecute() {
		return nil, errors.New(errors.CategoryDatabase, errors.CodeCircuitBreakerOpen, 
			"Database circuit breaker is open").Build()
	}

	start := time.Now()
	result, err := pm.db.ExecContext(ctx, query, args...)
	duration := time.Since(start)

	// AIDEV-NOTE: 250903170019 Record query metrics
	pm.recordQueryMetrics("exec", duration, err != nil)

	if err != nil {
		pm.healthCheck.CircuitBreaker.recordFailure()
		return nil, errors.WrapDatabase(err, "exec").WithContext("query", query)
	}

	pm.healthCheck.CircuitBreaker.recordSuccess()
	return result, nil
}

// QueryContextWithMonitoring executes a query with monitoring
func (pm *PoolManager) QueryContextWithMonitoring(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	if !pm.healthCheck.CircuitBreaker.canExecute() {
		return nil, errors.New(errors.CategoryDatabase, errors.CodeCircuitBreakerOpen,
			"Database circuit breaker is open").Build()
	}

	start := time.Now()
	rows, err := pm.db.QueryContext(ctx, query, args...)
	duration := time.Since(start)

	// AIDEV-NOTE: 250903170020 Record query metrics and detect slow queries
	pm.recordQueryMetrics("query", duration, err != nil)

	if err != nil {
		pm.healthCheck.CircuitBreaker.recordFailure()
		return nil, errors.WrapDatabase(err, "query").WithContext("query", query)
	}

	pm.healthCheck.CircuitBreaker.recordSuccess()
	return rows, nil
}

// recordQueryMetrics records metrics for query execution
func (pm *PoolManager) recordQueryMetrics(queryType string, duration time.Duration, failed bool) {
	pm.metrics.mu.Lock()
	defer pm.metrics.mu.Unlock()

	// AIDEV-NOTE: 250903170021 Track slow queries
	if duration > pm.config.SlowQueryThreshold {
		pm.metrics.SlowQueries++
		
		if pm.config.EnableSlowQueryLog {
			pm.logger.Warn("Slow query detected", logger.Fields{
				"query_type":  queryType,
				"duration_ms": duration.Milliseconds(),
				"threshold_ms": pm.config.SlowQueryThreshold.Milliseconds(),
			})
		}
	}

	// Update query latency metrics
	pm.metrics.QueryLatency[queryType] = duration

	if failed {
		pm.metrics.FailedConnections++
	}
}