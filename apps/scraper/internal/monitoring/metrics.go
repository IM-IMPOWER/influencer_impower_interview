// AIDEV-NOTE: 250903170008 Production metrics collection for KOL scraper service
// Prometheus metrics for observability and monitoring
package monitoring

import (
	"context"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"kol-scraper/pkg/logger"
)

// MetricsCollector handles Prometheus metrics collection
type MetricsCollector struct {
	// AIDEV-NOTE: HTTP request metrics
	httpRequestsTotal    *prometheus.CounterVec
	httpRequestDuration  *prometheus.HistogramVec
	httpActiveRequests   prometheus.Gauge

	// AIDEV-NOTE: Scraping metrics
	scrapingJobsTotal     *prometheus.CounterVec
	scrapingDuration      *prometheus.HistogramVec
	scrapingErrors        *prometheus.CounterVec
	activeScrapingJobs    prometheus.Gauge
	queueSize             prometheus.Gauge

	// AIDEV-NOTE: Database metrics
	dbConnectionsActive   prometheus.Gauge
	dbConnectionsIdle     prometheus.Gauge
	dbQueryDuration       *prometheus.HistogramVec
	dbErrors             *prometheus.CounterVec

	// AIDEV-NOTE: Circuit breaker metrics
	circuitBreakerState  *prometheus.GaugeVec
	circuitBreakerCounts *prometheus.CounterVec

	// AIDEV-NOTE: System metrics
	memoryUsage          prometheus.Gauge
	cpuUsage             prometheus.Gauge
	goroutineCount       prometheus.Gauge

	logger logger.Logger
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(log logger.Logger) *MetricsCollector {
	mc := &MetricsCollector{
		logger: log,
		
		// AIDEV-NOTE: Initialize HTTP metrics
		httpRequestsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "http_requests_total",
				Help: "Total number of HTTP requests",
			},
			[]string{"method", "endpoint", "status_code"},
		),
		
		httpRequestDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "http_request_duration_seconds",
				Help:    "HTTP request duration in seconds",
				Buckets: prometheus.DefBuckets,
			},
			[]string{"method", "endpoint", "status_code"},
		),
		
		httpActiveRequests: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "http_active_requests",
				Help: "Number of active HTTP requests",
			},
		),

		// AIDEV-NOTE: Initialize scraping metrics
		scrapingJobsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "scraping_jobs_total",
				Help: "Total number of scraping jobs",
			},
			[]string{"platform", "status"},
		),
		
		scrapingDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "scraping_duration_seconds",
				Help:    "Scraping job duration in seconds",
				Buckets: []float64{1, 5, 10, 30, 60, 300, 600},
			},
			[]string{"platform"},
		),
		
		scrapingErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "scraping_errors_total",
				Help: "Total number of scraping errors",
			},
			[]string{"platform", "error_type"},
		),
		
		activeScrapingJobs: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "active_scraping_jobs",
				Help: "Number of active scraping jobs",
			},
		),
		
		queueSize: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "job_queue_size",
				Help: "Current job queue size",
			},
		),

		// AIDEV-NOTE: Initialize database metrics
		dbConnectionsActive: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "db_connections_active",
				Help: "Number of active database connections",
			},
		),
		
		dbConnectionsIdle: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "db_connections_idle",
				Help: "Number of idle database connections",
			},
		),
		
		dbQueryDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "db_query_duration_seconds",
				Help:    "Database query duration in seconds",
				Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1},
			},
			[]string{"operation"},
		),
		
		dbErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "db_errors_total",
				Help: "Total number of database errors",
			},
			[]string{"operation", "error_type"},
		),

		// AIDEV-NOTE: Initialize circuit breaker metrics
		circuitBreakerState: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "circuit_breaker_state",
				Help: "Circuit breaker state (0=closed, 1=open, 2=half-open)",
			},
			[]string{"name"},
		),
		
		circuitBreakerCounts: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "circuit_breaker_events_total",
				Help: "Circuit breaker events",
			},
			[]string{"name", "event"},
		),

		// AIDEV-NOTE: Initialize system metrics
		memoryUsage: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "memory_usage_bytes",
				Help: "Current memory usage in bytes",
			},
		),
		
		cpuUsage: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "cpu_usage_percent",
				Help: "Current CPU usage percentage",
			},
		),
		
		goroutineCount: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "goroutines_count",
				Help: "Current number of goroutines",
			},
		),
	}

	// AIDEV-NOTE: Register all metrics with Prometheus
	prometheus.MustRegister(
		mc.httpRequestsTotal,
		mc.httpRequestDuration,
		mc.httpActiveRequests,
		mc.scrapingJobsTotal,
		mc.scrapingDuration,
		mc.scrapingErrors,
		mc.activeScrapingJobs,
		mc.queueSize,
		mc.dbConnectionsActive,
		mc.dbConnectionsIdle,
		mc.dbQueryDuration,
		mc.dbErrors,
		mc.circuitBreakerState,
		mc.circuitBreakerCounts,
		mc.memoryUsage,
		mc.cpuUsage,
		mc.goroutineCount,
	)

	return mc
}

// AIDEV-NOTE: HTTP middleware for request metrics
func (mc *MetricsCollector) HTTPMetricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		
		// Increment active requests
		mc.httpActiveRequests.Inc()
		defer mc.httpActiveRequests.Dec()

		// Process request
		c.Next()

		// Record metrics
		duration := time.Since(start).Seconds()
		statusCode := strconv.Itoa(c.Writer.Status())
		
		mc.httpRequestsTotal.WithLabelValues(
			c.Request.Method,
			c.FullPath(),
			statusCode,
		).Inc()
		
		mc.httpRequestDuration.WithLabelValues(
			c.Request.Method,
			c.FullPath(),
			statusCode,
		).Observe(duration)
	}
}

// AIDEV-NOTE: Scraping job metrics
func (mc *MetricsCollector) RecordScrapingJob(platform, status string, duration time.Duration) {
	mc.scrapingJobsTotal.WithLabelValues(platform, status).Inc()
	mc.scrapingDuration.WithLabelValues(platform).Observe(duration.Seconds())
}

func (mc *MetricsCollector) RecordScrapingError(platform, errorType string) {
	mc.scrapingErrors.WithLabelValues(platform, errorType).Inc()
}

func (mc *MetricsCollector) SetActiveScrapingJobs(count float64) {
	mc.activeScrapingJobs.Set(count)
}

func (mc *MetricsCollector) SetQueueSize(size float64) {
	mc.queueSize.Set(size)
}

// AIDEV-NOTE: Database metrics
func (mc *MetricsCollector) RecordDBQuery(operation string, duration time.Duration) {
	mc.dbQueryDuration.WithLabelValues(operation).Observe(duration.Seconds())
}

func (mc *MetricsCollector) RecordDBError(operation, errorType string) {
	mc.dbErrors.WithLabelValues(operation, errorType).Inc()
}

func (mc *MetricsCollector) SetDBConnections(active, idle float64) {
	mc.dbConnectionsActive.Set(active)
	mc.dbConnectionsIdle.Set(idle)
}

// AIDEV-NOTE: Circuit breaker metrics
func (mc *MetricsCollector) SetCircuitBreakerState(name string, state float64) {
	mc.circuitBreakerState.WithLabelValues(name).Set(state)
}

func (mc *MetricsCollector) RecordCircuitBreakerEvent(name, event string) {
	mc.circuitBreakerCounts.WithLabelValues(name, event).Inc()
}

// AIDEV-NOTE: System metrics collection
func (mc *MetricsCollector) UpdateSystemMetrics(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			mc.collectSystemMetrics()
		}
	}
}

func (mc *MetricsCollector) collectSystemMetrics() {
	// AIDEV-NOTE: This would need proper system metrics collection
	// For now, just update goroutine count
	mc.goroutineCount.Set(float64(runtime.NumGoroutine()))
	
	// TODO: Implement memory and CPU collection using appropriate libraries
	// like gopsutil or similar system monitoring libraries
}

// AIDEV-NOTE: Health check with detailed status
type HealthStatus struct {
	Status    string            `json:"status"`
	Timestamp int64             `json:"timestamp"`
	Version   string            `json:"version"`
	Checks    map[string]string `json:"checks"`
	Metrics   map[string]float64 `json:"metrics,omitempty"`
}

// GetHealthStatus returns detailed health status
func (mc *MetricsCollector) GetHealthStatus() HealthStatus {
	status := HealthStatus{
		Status:    "healthy",
		Timestamp: time.Now().Unix(),
		Version:   "1.0.0", // TODO: Get from build info
		Checks:    make(map[string]string),
		Metrics:   make(map[string]float64),
	}

	// AIDEV-NOTE: Add basic health checks
	status.Checks["database"] = "healthy" // TODO: Implement actual DB health check
	status.Checks["redis"] = "healthy"    // TODO: Implement actual Redis health check
	status.Checks["memory"] = "healthy"   // TODO: Check memory usage
	
	// AIDEV-NOTE: Add key metrics for monitoring dashboards
	status.Metrics["active_requests"] = float64(1) // TODO: Get from actual gauge
	status.Metrics["queue_size"] = float64(0)      // TODO: Get from actual queue
	status.Metrics["error_rate"] = 0.0             // TODO: Calculate from error metrics

	return status
}

// ServeMetrics serves Prometheus metrics endpoint
func (mc *MetricsCollector) ServeMetrics() http.Handler {
	return promhttp.Handler()
}