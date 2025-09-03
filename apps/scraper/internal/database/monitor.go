// AIDEV-NOTE: 250903170200 Database monitoring system for production visibility
// Provides comprehensive monitoring, alerting, and performance analytics
package database

import (
	"context"
	"database/sql"
	"fmt"
	"sync"
	"time"

	"github.com/jmoiron/sqlx"
	"kol-scraper/internal/errors"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: 250903170201 Database monitor with comprehensive metrics collection
type DatabaseMonitor struct {
	db              *sqlx.DB
	poolManager     *PoolManager
	logger          *logger.Logger
	metrics         *MonitorMetrics
	alerts          *AlertManager
	collectors      []MetricCollector
	config          *MonitorConfig
	stopCh          chan struct{}
	wg              sync.WaitGroup
	mu              sync.RWMutex
}

// AIDEV-NOTE: 250903170202 Monitoring configuration
type MonitorConfig struct {
	CollectionInterval    time.Duration `json:"collection_interval"`
	AlertingEnabled       bool          `json:"alerting_enabled"`
	SlowQueryThreshold    time.Duration `json:"slow_query_threshold"`
	DeadlockDetection     bool          `json:"deadlock_detection"`
	TableBloatMonitoring  bool          `json:"table_bloat_monitoring"`
	IndexUsageTracking    bool          `json:"index_usage_tracking"`
	ConnectionMonitoring  bool          `json:"connection_monitoring"`
	ExportMetrics         bool          `json:"export_metrics"`
	MetricsRetentionDays  int           `json:"metrics_retention_days"`
}

// AIDEV-NOTE: 250903170203 Comprehensive monitoring metrics
type MonitorMetrics struct {
	mu                    sync.RWMutex
	DatabaseSize          int64                 `json:"database_size_bytes"`
	TableSizes            map[string]int64      `json:"table_sizes"`
	IndexSizes            map[string]int64      `json:"index_sizes"`
	IndexUsage            map[string]*IndexStat `json:"index_usage"`
	SlowQueries           []*SlowQueryStat      `json:"slow_queries"`
	Deadlocks             int64                 `json:"deadlocks"`
	Transactions          *TransactionStats     `json:"transactions"`
	ConnectionStats       *ConnectionStats      `json:"connection_stats"`
	ReplicationLag        time.Duration         `json:"replication_lag"`
	CacheHitRatio         float64              `json:"cache_hit_ratio"`
	LastCollected         time.Time            `json:"last_collected"`
	AlertsTriggered       int64                `json:"alerts_triggered"`
}

// AIDEV-NOTE: 250903170204 Database statistics structures
type IndexStat struct {
	SchemaName    string `db:"schemaname" json:"schema_name"`
	TableName     string `db:"tablename" json:"table_name"`
	IndexName     string `db:"indexname" json:"index_name"`
	TupleReads    int64  `db:"idx_tup_read" json:"tuple_reads"`
	TupleFetches  int64  `db:"idx_tup_fetch" json:"tuple_fetches"`
	Size          int64  `db:"size" json:"size"`
	LastUsed      *time.Time `json:"last_used,omitempty"`
}

type SlowQueryStat struct {
	Query         string        `db:"query" json:"query"`
	Calls         int64         `db:"calls" json:"calls"`
	TotalTime     time.Duration `db:"total_time" json:"total_time"`
	MeanTime      time.Duration `db:"mean_time" json:"mean_time"`
	MaxTime       time.Duration `db:"max_time" json:"max_time"`
	Rows          int64         `db:"rows" json:"rows"`
	FirstSeen     time.Time     `json:"first_seen"`
	LastSeen      time.Time     `json:"last_seen"`
}

type TransactionStats struct {
	Committed    int64   `db:"xact_commit" json:"committed"`
	RolledBack   int64   `db:"xact_rollback" json:"rolled_back"`
	DeadLocks    int64   `db:"deadlocks" json:"deadlocks"`
	ConflictRate float64 `json:"conflict_rate"`
}

type ConnectionStats struct {
	Active      int `db:"active" json:"active"`
	Idle        int `db:"idle" json:"idle"`
	IdleInTrans int `db:"idle_in_transaction" json:"idle_in_transaction"`
	Waiting     int `db:"waiting" json:"waiting"`
	Total       int `json:"total"`
}

// AIDEV-NOTE: 250903170205 Alert management system
type AlertManager struct {
	mu        sync.RWMutex
	rules     []*AlertRule
	active    map[string]*ActiveAlert
	handlers  []AlertHandler
	logger    *logger.Logger
}

type AlertRule struct {
	Name        string                 `json:"name"`
	Condition   func(*MonitorMetrics) bool `json:"-"`
	Severity    AlertSeverity          `json:"severity"`
	Message     string                 `json:"message"`
	Cooldown    time.Duration          `json:"cooldown"`
	LastFired   time.Time             `json:"last_fired"`
	Enabled     bool                  `json:"enabled"`
}

type ActiveAlert struct {
	Rule      *AlertRule    `json:"rule"`
	StartTime time.Time     `json:"start_time"`
	Message   string        `json:"message"`
	Resolved  bool         `json:"resolved"`
}

type AlertSeverity string

const (
	SeverityInfo     AlertSeverity = "info"
	SeverityWarning  AlertSeverity = "warning"
	SeverityCritical AlertSeverity = "critical"
)

type AlertHandler interface {
	HandleAlert(alert *ActiveAlert) error
}

// AIDEV-NOTE: 250903170206 Metric collectors interface
type MetricCollector interface {
	Collect(ctx context.Context, db *sqlx.DB) (interface{}, error)
	Name() string
	Interval() time.Duration
}

// NewDatabaseMonitor creates a new database monitoring system
func NewDatabaseMonitor(db *sqlx.DB, poolManager *PoolManager, logger *logger.Logger) *DatabaseMonitor {
	config := &MonitorConfig{
		CollectionInterval:    30 * time.Second,
		AlertingEnabled:       true,
		SlowQueryThreshold:    1 * time.Second,
		DeadlockDetection:     true,
		TableBloatMonitoring:  true,
		IndexUsageTracking:    true,
		ConnectionMonitoring:  true,
		ExportMetrics:         true,
		MetricsRetentionDays:  30,
	}

	dm := &DatabaseMonitor{
		db:          db,
		poolManager: poolManager,
		logger:      logger,
		config:      config,
		metrics: &MonitorMetrics{
			TableSizes:      make(map[string]int64),
			IndexSizes:      make(map[string]int64),
			IndexUsage:      make(map[string]*IndexStat),
			SlowQueries:     make([]*SlowQueryStat, 0),
			LastCollected:   time.Now(),
		},
		alerts: NewAlertManager(logger),
		stopCh: make(chan struct{}),
	}

	// AIDEV-NOTE: 250903170207 Initialize metric collectors
	dm.initializeCollectors()

	// AIDEV-NOTE: 250903170208 Set up default alert rules
	dm.setupDefaultAlerts()

	return dm
}

// initializeCollectors sets up various metric collectors
func (dm *DatabaseMonitor) initializeCollectors() {
	dm.collectors = []MetricCollector{
		&TableSizeCollector{},
		&IndexUsageCollector{},
		&SlowQueryCollector{threshold: dm.config.SlowQueryThreshold},
		&ConnectionStatsCollector{},
		&TransactionStatsCollector{},
		&DeadlockCollector{},
	}

	dm.logger.Info("Database metric collectors initialized", logger.Fields{
		"collectors": len(dm.collectors),
	})
}

// setupDefaultAlerts configures default monitoring alerts
func (dm *DatabaseMonitor) setupDefaultAlerts() {
	rules := []*AlertRule{
		{
			Name:     "High Connection Usage",
			Severity: SeverityWarning,
			Message:  "Database connection pool usage is above 80%",
			Cooldown: 5 * time.Minute,
			Enabled:  true,
			Condition: func(m *MonitorMetrics) bool {
				if m.ConnectionStats == nil {
					return false
				}
				poolStats := dm.poolManager.GetMetrics()
				return float64(poolStats.ConnectionsInUse)/float64(dm.config.CollectionInterval.Seconds()) > 0.8
			},
		},
		{
			Name:     "Slow Query Detection",
			Severity: SeverityWarning,
			Message:  "Multiple slow queries detected",
			Cooldown: 2 * time.Minute,
			Enabled:  true,
			Condition: func(m *MonitorMetrics) bool {
				return len(m.SlowQueries) > 5
			},
		},
		{
			Name:     "Database Size Growth",
			Severity: SeverityInfo,
			Message:  "Database size has grown significantly",
			Cooldown: 1 * time.Hour,
			Enabled:  true,
			Condition: func(m *MonitorMetrics) bool {
				// Alert if database is over 1GB
				return m.DatabaseSize > 1024*1024*1024
			},
		},
		{
			Name:     "Deadlock Detection",
			Severity: SeverityCritical,
			Message:  "Database deadlocks detected",
			Cooldown: 1 * time.Minute,
			Enabled:  true,
			Condition: func(m *MonitorMetrics) bool {
				return m.Deadlocks > 0
			},
		},
		{
			Name:     "Low Cache Hit Ratio",
			Severity: SeverityWarning,
			Message:  "Database cache hit ratio is below optimal threshold",
			Cooldown: 10 * time.Minute,
			Enabled:  true,
			Condition: func(m *MonitorMetrics) bool {
				return m.CacheHitRatio < 0.95 // Below 95%
			},
		},
	}

	for _, rule := range rules {
		dm.alerts.AddRule(rule)
	}

	dm.logger.Info("Default alert rules configured", logger.Fields{
		"rules": len(rules),
	})
}

// Start begins the monitoring process
func (dm *DatabaseMonitor) Start(ctx context.Context) error {
	dm.logger.Info("Starting database monitor")

	dm.wg.Add(1)
	go dm.monitoringLoop(ctx)

	return nil
}

// monitoringLoop runs the main monitoring collection loop
func (dm *DatabaseMonitor) monitoringLoop(ctx context.Context) {
	defer dm.wg.Done()

	ticker := time.NewTicker(dm.config.CollectionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := dm.collectMetrics(ctx); err != nil {
				dm.logger.ErrorLog(errors.WrapDatabase(err, "collect_metrics"),
					"metrics_collection_failed", logger.Fields{
						"collection_interval": dm.config.CollectionInterval,
					})
			}

			// AIDEV-NOTE: 250903170209 Check alerts after metrics collection
			if dm.config.AlertingEnabled {
				dm.alerts.CheckAlerts(dm.metrics)
			}

		case <-dm.stopCh:
			return
		case <-ctx.Done():
			return
		}
	}
}

// collectMetrics runs all metric collectors
func (dm *DatabaseMonitor) collectMetrics(ctx context.Context) error {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	start := time.Now()
	errors := make([]error, 0)

	// AIDEV-NOTE: 250903170210 Run all metric collectors
	for _, collector := range dm.collectors {
		collectorCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		
		result, err := collector.Collect(collectorCtx, dm.db)
		cancel()

		if err != nil {
			errors = append(errors, fmt.Errorf("collector %s failed: %w", collector.Name(), err))
			continue
		}

		// AIDEV-NOTE: 250903170211 Update metrics based on collector results
		dm.updateMetricsFromCollector(collector.Name(), result)
	}

	dm.metrics.LastCollected = time.Now()
	
	duration := time.Since(start)
	dm.logger.Debug("Metrics collection completed", logger.Fields{
		"duration_ms": duration.Milliseconds(),
		"errors":      len(errors),
	})

	if len(errors) > 0 {
		return fmt.Errorf("metric collection had %d errors", len(errors))
	}

	return nil
}

// updateMetricsFromCollector updates metrics based on collector results
func (dm *DatabaseMonitor) updateMetricsFromCollector(collectorName string, result interface{}) {
	switch collectorName {
	case "table_sizes":
		if sizes, ok := result.(map[string]int64); ok {
			dm.metrics.TableSizes = sizes
			
			// Calculate total database size
			var total int64
			for _, size := range sizes {
				total += size
			}
			dm.metrics.DatabaseSize = total
		}

	case "index_usage":
		if usage, ok := result.(map[string]*IndexStat); ok {
			dm.metrics.IndexUsage = usage
		}

	case "slow_queries":
		if queries, ok := result.([]*SlowQueryStat); ok {
			dm.metrics.SlowQueries = queries
		}

	case "connection_stats":
		if stats, ok := result.(*ConnectionStats); ok {
			dm.metrics.ConnectionStats = stats
		}

	case "transaction_stats":
		if stats, ok := result.(*TransactionStats); ok {
			dm.metrics.Transactions = stats
		}

	case "deadlocks":
		if count, ok := result.(int64); ok {
			dm.metrics.Deadlocks = count
		}
	}
}

// GetMetrics returns current monitoring metrics
func (dm *DatabaseMonitor) GetMetrics() *MonitorMetrics {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	// Create a deep copy to avoid race conditions
	copy := &MonitorMetrics{
		DatabaseSize:     dm.metrics.DatabaseSize,
		TableSizes:       make(map[string]int64),
		IndexSizes:       make(map[string]int64),
		IndexUsage:       make(map[string]*IndexStat),
		SlowQueries:      make([]*SlowQueryStat, len(dm.metrics.SlowQueries)),
		Deadlocks:        dm.metrics.Deadlocks,
		ReplicationLag:   dm.metrics.ReplicationLag,
		CacheHitRatio:    dm.metrics.CacheHitRatio,
		LastCollected:    dm.metrics.LastCollected,
		AlertsTriggered:  dm.metrics.AlertsTriggered,
	}

	// Copy maps
	for k, v := range dm.metrics.TableSizes {
		copy.TableSizes[k] = v
	}
	for k, v := range dm.metrics.IndexSizes {
		copy.IndexSizes[k] = v
	}
	for k, v := range dm.metrics.IndexUsage {
		indexCopy := *v
		copy.IndexUsage[k] = &indexCopy
	}

	// Copy slice
	copy(copy.SlowQueries, dm.metrics.SlowQueries)

	// Copy structs
	if dm.metrics.Transactions != nil {
		transCopy := *dm.metrics.Transactions
		copy.Transactions = &transCopy
	}
	if dm.metrics.ConnectionStats != nil {
		connCopy := *dm.metrics.ConnectionStats
		copy.ConnectionStats = &connCopy
	}

	return copy
}

// GetAlerts returns current active alerts
func (dm *DatabaseMonitor) GetAlerts() []*ActiveAlert {
	return dm.alerts.GetActiveAlerts()
}

// Stop gracefully shuts down the monitoring system
func (dm *DatabaseMonitor) Stop() error {
	dm.logger.Info("Stopping database monitor")

	close(dm.stopCh)
	dm.wg.Wait()

	dm.logger.Info("Database monitor stopped")
	return nil
}

// AIDEV-NOTE: 250903170212 Alert manager implementation
func NewAlertManager(logger *logger.Logger) *AlertManager {
	return &AlertManager{
		rules:    make([]*AlertRule, 0),
		active:   make(map[string]*ActiveAlert),
		handlers: make([]AlertHandler, 0),
		logger:   logger,
	}
}

func (am *AlertManager) AddRule(rule *AlertRule) {
	am.mu.Lock()
	defer am.mu.Unlock()

	am.rules = append(am.rules, rule)
}

func (am *AlertManager) AddHandler(handler AlertHandler) {
	am.mu.Lock()
	defer am.mu.Unlock()

	am.handlers = append(am.handlers, handler)
}

func (am *AlertManager) CheckAlerts(metrics *MonitorMetrics) {
	am.mu.Lock()
	defer am.mu.Unlock()

	for _, rule := range am.rules {
		if !rule.Enabled {
			continue
		}

		// Check if rule condition is met
		if rule.Condition(metrics) {
			// Check cooldown
			if time.Since(rule.LastFired) < rule.Cooldown {
				continue
			}

			// Create or update alert
			alertKey := rule.Name
			alert, exists := am.active[alertKey]
			
			if !exists {
				alert = &ActiveAlert{
					Rule:      rule,
					StartTime: time.Now(),
					Message:   rule.Message,
					Resolved:  false,
				}
				am.active[alertKey] = alert

				// Fire alert to handlers
				am.fireAlert(alert)
				rule.LastFired = time.Now()
			}
		} else {
			// Resolve alert if it exists
			alertKey := rule.Name
			if alert, exists := am.active[alertKey]; exists && !alert.Resolved {
				alert.Resolved = true
				am.resolveAlert(alert)
			}
		}
	}
}

func (am *AlertManager) fireAlert(alert *ActiveAlert) {
	am.logger.Warn("Database alert triggered", logger.Fields{
		"alert_name": alert.Rule.Name,
		"severity":   string(alert.Rule.Severity),
		"message":    alert.Message,
	})

	for _, handler := range am.handlers {
		if err := handler.HandleAlert(alert); err != nil {
			am.logger.ErrorLog(errors.WrapIntegration(err, "alert_handler"),
				"alert_handler_failed", logger.Fields{
					"alert_name": alert.Rule.Name,
				})
		}
	}
}

func (am *AlertManager) resolveAlert(alert *ActiveAlert) {
	am.logger.Info("Database alert resolved", logger.Fields{
		"alert_name": alert.Rule.Name,
		"duration":   time.Since(alert.StartTime).String(),
	})
}

func (am *AlertManager) GetActiveAlerts() []*ActiveAlert {
	am.mu.RLock()
	defer am.mu.RUnlock()

	alerts := make([]*ActiveAlert, 0, len(am.active))
	for _, alert := range am.active {
		if !alert.Resolved {
			alerts = append(alerts, alert)
		}
	}

	return alerts
}

// AIDEV-NOTE: 250903170213 Default log alert handler
type LogAlertHandler struct {
	logger *logger.Logger
}

func NewLogAlertHandler(logger *logger.Logger) *LogAlertHandler {
	return &LogAlertHandler{logger: logger}
}

func (h *LogAlertHandler) HandleAlert(alert *ActiveAlert) error {
	fields := logger.Fields{
		"alert_name":  alert.Rule.Name,
		"severity":    string(alert.Rule.Severity),
		"message":     alert.Message,
		"start_time":  alert.StartTime,
	}

	switch alert.Rule.Severity {
	case SeverityCritical:
		h.logger.Error("CRITICAL DATABASE ALERT", fields)
	case SeverityWarning:
		h.logger.Warn("Database warning alert", fields)
	case SeverityInfo:
		h.logger.Info("Database info alert", fields)
	}

	return nil
}