// AIDEV-NOTE: 250903170009 Comprehensive health checking for production deployment
// Health checks for database, Redis, external services, and system resources
package monitoring

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"runtime"
	"time"

	"github.com/redis/go-redis/v9"
	"kol-scraper/pkg/logger"
)

// HealthChecker performs comprehensive health checks
type HealthChecker struct {
	db     *sql.DB
	redis  *redis.Client
	logger logger.Logger
	config HealthConfig
}

// HealthConfig holds health check configuration
type HealthConfig struct {
	Timeout           time.Duration
	DatabaseTimeout   time.Duration
	RedisTimeout      time.Duration
	ExternalTimeout   time.Duration
	MemoryThreshold   uint64 // MB
	GoroutineThreshold int
}

// HealthCheckResult represents individual check result
type HealthCheckResult struct {
	Name       string        `json:"name"`
	Status     string        `json:"status"`
	Message    string        `json:"message,omitempty"`
	Duration   time.Duration `json:"duration"`
	Timestamp  time.Time     `json:"timestamp"`
	Critical   bool          `json:"critical"`
}

// OverallHealth represents the complete health status
type OverallHealth struct {
	Status      string                         `json:"status"`
	Timestamp   time.Time                     `json:"timestamp"`
	Version     string                        `json:"version"`
	Uptime      time.Duration                 `json:"uptime"`
	Checks      map[string]HealthCheckResult  `json:"checks"`
	SystemInfo  SystemInfo                    `json:"system_info"`
}

// SystemInfo contains system resource information
type SystemInfo struct {
	MemoryUsedMB    uint64  `json:"memory_used_mb"`
	MemoryTotalMB   uint64  `json:"memory_total_mb"`
	GoroutineCount  int     `json:"goroutine_count"`
	CPUCount        int     `json:"cpu_count"`
	GoVersion       string  `json:"go_version"`
}

var (
	startTime = time.Now()
)

// NewHealthChecker creates a new health checker
func NewHealthChecker(db *sql.DB, redis *redis.Client, logger logger.Logger) *HealthChecker {
	return &HealthChecker{
		db:     db,
		redis:  redis,
		logger: logger,
		config: HealthConfig{
			Timeout:           30 * time.Second,
			DatabaseTimeout:   5 * time.Second,
			RedisTimeout:      3 * time.Second,
			ExternalTimeout:   10 * time.Second,
			MemoryThreshold:   1024, // 1GB
			GoroutineThreshold: 10000,
		},
	}
}

// AIDEV-NOTE: Perform all health checks
func (hc *HealthChecker) CheckHealth(ctx context.Context) OverallHealth {
	checks := make(map[string]HealthCheckResult)
	overallStatus := "healthy"
	
	// AIDEV-NOTE: Run all health checks concurrently
	checkFunctions := map[string]func(context.Context) HealthCheckResult{
		"database":      hc.checkDatabase,
		"redis":         hc.checkRedis,
		"memory":        hc.checkMemory,
		"goroutines":    hc.checkGoroutines,
		"disk_space":    hc.checkDiskSpace,
		"fastapi_service": hc.checkFastAPIService,
	}
	
	// AIDEV-NOTE: Execute checks with timeout
	checkCtx, cancel := context.WithTimeout(ctx, hc.config.Timeout)
	defer cancel()
	
	resultChan := make(chan struct {
		name   string
		result HealthCheckResult
	}, len(checkFunctions))
	
	// Start all checks concurrently
	for name, checkFunc := range checkFunctions {
		go func(n string, f func(context.Context) HealthCheckResult) {
			result := f(checkCtx)
			resultChan <- struct {
				name   string
				result HealthCheckResult
			}{n, result}
		}(name, checkFunc)
	}
	
	// Collect results
	for i := 0; i < len(checkFunctions); i++ {
		select {
		case result := <-resultChan:
			checks[result.name] = result.result
			
			// AIDEV-NOTE: Update overall status based on critical checks
			if result.result.Critical && result.result.Status != "healthy" {
				overallStatus = "unhealthy"
			} else if overallStatus == "healthy" && result.result.Status == "degraded" {
				overallStatus = "degraded"
			}
			
		case <-checkCtx.Done():
			hc.logger.Warn("Health check timeout", "timeout", hc.config.Timeout)
			overallStatus = "timeout"
			break
		}
	}
	
	return OverallHealth{
		Status:     overallStatus,
		Timestamp:  time.Now(),
		Version:    "1.0.0", // TODO: Get from build info
		Uptime:     time.Since(startTime),
		Checks:     checks,
		SystemInfo: hc.getSystemInfo(),
	}
}

// AIDEV-NOTE: Database connectivity check
func (hc *HealthChecker) checkDatabase(ctx context.Context) HealthCheckResult {
	start := time.Now()
	result := HealthCheckResult{
		Name:      "database",
		Timestamp: start,
		Critical:  true,
	}
	
	if hc.db == nil {
		result.Status = "unhealthy"
		result.Message = "Database connection not initialized"
		result.Duration = time.Since(start)
		return result
	}
	
	// Create timeout context for database check
	dbCtx, cancel := context.WithTimeout(ctx, hc.config.DatabaseTimeout)
	defer cancel()
	
	// AIDEV-NOTE: Test connection with simple query
	err := hc.db.PingContext(dbCtx)
	if err != nil {
		result.Status = "unhealthy"
		result.Message = fmt.Sprintf("Database ping failed: %v", err)
		result.Duration = time.Since(start)
		return result
	}
	
	// AIDEV-NOTE: Test query execution
	var count int
	err = hc.db.QueryRowContext(dbCtx, "SELECT 1").Scan(&count)
	if err != nil {
		result.Status = "degraded"
		result.Message = fmt.Sprintf("Database query test failed: %v", err)
		result.Duration = time.Since(start)
		return result
	}
	
	result.Status = "healthy"
	result.Message = "Database connection successful"
	result.Duration = time.Since(start)
	return result
}

// AIDEV-NOTE: Redis connectivity check
func (hc *HealthChecker) checkRedis(ctx context.Context) HealthCheckResult {
	start := time.Now()
	result := HealthCheckResult{
		Name:      "redis",
		Timestamp: start,
		Critical:  true,
	}
	
	if hc.redis == nil {
		result.Status = "unhealthy"
		result.Message = "Redis connection not initialized"
		result.Duration = time.Since(start)
		return result
	}
	
	// Create timeout context for Redis check
	redisCtx, cancel := context.WithTimeout(ctx, hc.config.RedisTimeout)
	defer cancel()
	
	// AIDEV-NOTE: Test Redis connectivity
	_, err := hc.redis.Ping(redisCtx).Result()
	if err != nil {
		result.Status = "unhealthy"
		result.Message = fmt.Sprintf("Redis ping failed: %v", err)
		result.Duration = time.Since(start)
		return result
	}
	
	// AIDEV-NOTE: Test Redis operations
	testKey := fmt.Sprintf("health_check_%d", time.Now().Unix())
	err = hc.redis.Set(redisCtx, testKey, "test", time.Second).Err()
	if err != nil {
		result.Status = "degraded"
		result.Message = fmt.Sprintf("Redis write test failed: %v", err)
		result.Duration = time.Since(start)
		return result
	}
	
	// Cleanup test key
	hc.redis.Del(redisCtx, testKey)
	
	result.Status = "healthy"
	result.Message = "Redis connection successful"
	result.Duration = time.Since(start)
	return result
}

// AIDEV-NOTE: Memory usage check
func (hc *HealthChecker) checkMemory(ctx context.Context) HealthCheckResult {
	start := time.Now()
	result := HealthCheckResult{
		Name:      "memory",
		Timestamp: start,
		Critical:  false,
	}
	
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	usedMB := m.Alloc / 1024 / 1024
	
	if usedMB > hc.config.MemoryThreshold {
		result.Status = "degraded"
		result.Message = fmt.Sprintf("High memory usage: %d MB (threshold: %d MB)", usedMB, hc.config.MemoryThreshold)
	} else {
		result.Status = "healthy"
		result.Message = fmt.Sprintf("Memory usage: %d MB", usedMB)
	}
	
	result.Duration = time.Since(start)
	return result
}

// AIDEV-NOTE: Goroutine count check
func (hc *HealthChecker) checkGoroutines(ctx context.Context) HealthCheckResult {
	start := time.Now()
	result := HealthCheckResult{
		Name:      "goroutines",
		Timestamp: start,
		Critical:  false,
	}
	
	count := runtime.NumGoroutine()
	
	if count > hc.config.GoroutineThreshold {
		result.Status = "degraded"
		result.Message = fmt.Sprintf("High goroutine count: %d (threshold: %d)", count, hc.config.GoroutineThreshold)
	} else {
		result.Status = "healthy"
		result.Message = fmt.Sprintf("Goroutine count: %d", count)
	}
	
	result.Duration = time.Since(start)
	return result
}

// AIDEV-NOTE: Disk space check (simplified)
func (hc *HealthChecker) checkDiskSpace(ctx context.Context) HealthCheckResult {
	start := time.Now()
	result := HealthCheckResult{
		Name:      "disk_space",
		Timestamp: start,
		Critical:  false,
		Status:    "healthy",
		Message:   "Disk space check not implemented",
		Duration:  time.Since(start),
	}
	
	// TODO: Implement actual disk space check using syscall or third-party library
	return result
}

// AIDEV-NOTE: External FastAPI service check
func (hc *HealthChecker) checkFastAPIService(ctx context.Context) HealthCheckResult {
	start := time.Now()
	result := HealthCheckResult{
		Name:      "fastapi_service",
		Timestamp: start,
		Critical:  false,
	}
	
	// TODO: Get FastAPI URL from config
	fastapiURL := "http://localhost:8000/api/health"
	
	client := &http.Client{
		Timeout: hc.config.ExternalTimeout,
	}
	
	req, err := http.NewRequestWithContext(ctx, "GET", fastapiURL, nil)
	if err != nil {
		result.Status = "unhealthy"
		result.Message = fmt.Sprintf("Failed to create request: %v", err)
		result.Duration = time.Since(start)
		return result
	}
	
	resp, err := client.Do(req)
	if err != nil {
		result.Status = "unhealthy"
		result.Message = fmt.Sprintf("FastAPI service unreachable: %v", err)
		result.Duration = time.Since(start)
		return result
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		result.Status = "degraded"
		result.Message = fmt.Sprintf("FastAPI service returned status: %d", resp.StatusCode)
	} else {
		result.Status = "healthy"
		result.Message = "FastAPI service accessible"
	}
	
	result.Duration = time.Since(start)
	return result
}

// AIDEV-NOTE: Get system information
func (hc *HealthChecker) getSystemInfo() SystemInfo {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	return SystemInfo{
		MemoryUsedMB:   m.Alloc / 1024 / 1024,
		MemoryTotalMB:  m.Sys / 1024 / 1024,
		GoroutineCount: runtime.NumGoroutine(),
		CPUCount:       runtime.NumCPU(),
		GoVersion:      runtime.Version(),
	}
}

// AIDEV-NOTE: HTTP handler for health endpoint
func (hc *HealthChecker) HealthHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		health := hc.CheckHealth(ctx)
		
		// Set appropriate status code
		statusCode := http.StatusOK
		if health.Status == "unhealthy" {
			statusCode = http.StatusServiceUnavailable
		} else if health.Status == "degraded" {
			statusCode = http.StatusMultipleChoices // 300 for degraded
		}
		
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(statusCode)
		
		if err := json.NewEncoder(w).Encode(health); err != nil {
			hc.logger.Error("Failed to encode health response", "error", err)
			http.Error(w, "Internal server error", http.StatusInternalServerError)
		}
	}
}

// AIDEV-NOTE: Simple readiness check for Kubernetes
func (hc *HealthChecker) ReadinessHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		
		// Only check critical services for readiness
		checks := []func(context.Context) HealthCheckResult{
			hc.checkDatabase,
			hc.checkRedis,
		}
		
		for _, check := range checks {
			result := check(ctx)
			if result.Critical && result.Status != "healthy" {
				w.WriteHeader(http.StatusServiceUnavailable)
				json.NewEncoder(w).Encode(map[string]string{
					"status": "not_ready",
					"reason": result.Message,
				})
				return
			}
		}
		
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]string{
			"status": "ready",
		})
	}
}

// AIDEV-NOTE: Simple liveness check for Kubernetes
func (hc *HealthChecker) LivenessHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "alive",
			"timestamp": time.Now().Unix(),
			"uptime": time.Since(startTime).Seconds(),
		})
	}
}