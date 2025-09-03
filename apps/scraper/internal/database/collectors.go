// AIDEV-NOTE: 250903170300 Database metric collectors for comprehensive monitoring
// Implements specialized collectors for different aspects of database performance
package database

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/jmoiron/sqlx"
	"kol-scraper/internal/errors"
)

// AIDEV-NOTE: 250903170301 Table size collector for storage monitoring
type TableSizeCollector struct{}

func (c *TableSizeCollector) Name() string {
	return "table_sizes"
}

func (c *TableSizeCollector) Interval() time.Duration {
	return 5 * time.Minute
}

func (c *TableSizeCollector) Collect(ctx context.Context, db *sqlx.DB) (interface{}, error) {
	query := `
		SELECT 
			schemaname,
			tablename,
			pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
		FROM pg_tables 
		WHERE schemaname = 'public'
		ORDER BY size_bytes DESC`

	type tableSize struct {
		SchemaName string `db:"schemaname"`
		TableName  string `db:"tablename"`
		SizeBytes  int64  `db:"size_bytes"`
	}

	var results []tableSize
	err := db.SelectContext(ctx, &results, query)
	if err != nil {
		return nil, errors.WrapDatabase(err, "collect_table_sizes")
	}

	sizes := make(map[string]int64)
	for _, result := range results {
		key := fmt.Sprintf("%s.%s", result.SchemaName, result.TableName)
		sizes[key] = result.SizeBytes
	}

	return sizes, nil
}

// AIDEV-NOTE: 250903170302 Index usage collector for optimization insights
type IndexUsageCollector struct{}

func (c *IndexUsageCollector) Name() string {
	return "index_usage"
}

func (c *IndexUsageCollector) Interval() time.Duration {
	return 2 * time.Minute
}

func (c *IndexUsageCollector) Collect(ctx context.Context, db *sqlx.DB) (interface{}, error) {
	query := `
		SELECT 
			schemaname,
			tablename,
			indexname,
			idx_tup_read,
			idx_tup_fetch,
			pg_relation_size(indexrelid) as size
		FROM pg_stat_user_indexes 
		WHERE schemaname = 'public'
		ORDER BY idx_tup_read DESC`

	var results []IndexStat
	err := db.SelectContext(ctx, &results, query)
	if err != nil {
		return nil, errors.WrapDatabase(err, "collect_index_usage")
	}

	usage := make(map[string]*IndexStat)
	for i := range results {
		key := fmt.Sprintf("%s.%s.%s", results[i].SchemaName, results[i].TableName, results[i].IndexName)
		usage[key] = &results[i]
	}

	return usage, nil
}

// AIDEV-NOTE: 250903170303 Slow query collector for performance analysis
type SlowQueryCollector struct {
	threshold time.Duration
}

func (c *SlowQueryCollector) Name() string {
	return "slow_queries"
}

func (c *SlowQueryCollector) Interval() time.Duration {
	return 1 * time.Minute
}

func (c *SlowQueryCollector) Collect(ctx context.Context, db *sqlx.DB) (interface{}, error) {
	// AIDEV-NOTE: 250903170304 Check if pg_stat_statements extension is available
	var extensionExists bool
	checkExtQuery := `
		SELECT EXISTS (
			SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
		)`
	
	err := db.GetContext(ctx, &extensionExists, checkExtQuery)
	if err != nil {
		return []*SlowQueryStat{}, nil // Return empty if extension check fails
	}

	if !extensionExists {
		return []*SlowQueryStat{}, nil // Return empty if extension not available
	}

	query := `
		SELECT 
			query,
			calls,
			total_time * 1000 as total_time, -- Convert to microseconds
			mean_time * 1000 as mean_time,   -- Convert to microseconds
			max_time * 1000 as max_time,     -- Convert to microseconds
			rows
		FROM pg_stat_statements 
		WHERE mean_time > $1 -- Threshold in milliseconds
		  AND query NOT LIKE '%pg_stat_statements%'
		  AND query NOT LIKE '%information_schema%'
		  AND query NOT LIKE '%pg_catalog%'
		ORDER BY mean_time DESC 
		LIMIT 20`

	type slowQueryResult struct {
		Query     string  `db:"query"`
		Calls     int64   `db:"calls"`
		TotalTime float64 `db:"total_time"`
		MeanTime  float64 `db:"mean_time"`
		MaxTime   float64 `db:"max_time"`
		Rows      int64   `db:"rows"`
	}

	var results []slowQueryResult
	thresholdMs := float64(c.threshold.Nanoseconds()) / 1000000 // Convert to milliseconds
	err = db.SelectContext(ctx, &results, query, thresholdMs)
	if err != nil {
		return nil, errors.WrapDatabase(err, "collect_slow_queries")
	}

	slowQueries := make([]*SlowQueryStat, 0, len(results))
	now := time.Now()

	for _, result := range results {
		// AIDEV-NOTE: 250903170305 Clean and truncate long queries
		cleanQuery := strings.TrimSpace(result.Query)
		if len(cleanQuery) > 200 {
			cleanQuery = cleanQuery[:200] + "..."
		}

		slowQueries = append(slowQueries, &SlowQueryStat{
			Query:     cleanQuery,
			Calls:     result.Calls,
			TotalTime: time.Duration(result.TotalTime) * time.Microsecond,
			MeanTime:  time.Duration(result.MeanTime) * time.Microsecond,
			MaxTime:   time.Duration(result.MaxTime) * time.Microsecond,
			Rows:      result.Rows,
			LastSeen:  now,
		})
	}

	return slowQueries, nil
}

// AIDEV-NOTE: 250903170306 Connection statistics collector
type ConnectionStatsCollector struct{}

func (c *ConnectionStatsCollector) Name() string {
	return "connection_stats"
}

func (c *ConnectionStatsCollector) Interval() time.Duration {
	return 30 * time.Second
}

func (c *ConnectionStatsCollector) Collect(ctx context.Context, db *sqlx.DB) (interface{}, error) {
	query := `
		SELECT 
			state,
			count(*) as count
		FROM pg_stat_activity 
		WHERE datname = current_database()
		GROUP BY state`

	type connectionResult struct {
		State string `db:"state"`
		Count int    `db:"count"`
	}

	var results []connectionResult
	err := db.SelectContext(ctx, &results, query)
	if err != nil {
		return nil, errors.WrapDatabase(err, "collect_connection_stats")
	}

	stats := &ConnectionStats{}
	for _, result := range results {
		switch result.State {
		case "active":
			stats.Active = result.Count
		case "idle":
			stats.Idle = result.Count
		case "idle in transaction":
			stats.IdleInTrans = result.Count
		case "waiting":
			stats.Waiting = result.Count
		}
		stats.Total += result.Count
	}

	return stats, nil
}

// AIDEV-NOTE: 250903170307 Transaction statistics collector
type TransactionStatsCollector struct{}

func (c *TransactionStatsCollector) Name() string {
	return "transaction_stats"
}

func (c *TransactionStatsCollector) Interval() time.Duration {
	return 1 * time.Minute
}

func (c *TransactionStatsCollector) Collect(ctx context.Context, db *sqlx.DB) (interface{}, error) {
	query := `
		SELECT 
			xact_commit,
			xact_rollback,
			deadlocks,
			conflicts
		FROM pg_stat_database 
		WHERE datname = current_database()`

	type transactionResult struct {
		Committed  int64 `db:"xact_commit"`
		RolledBack int64 `db:"xact_rollback"`
		Deadlocks  int64 `db:"deadlocks"`
		Conflicts  int64 `db:"conflicts"`
	}

	var result transactionResult
	err := db.GetContext(ctx, &result, query)
	if err != nil {
		return nil, errors.WrapDatabase(err, "collect_transaction_stats")
	}

	stats := &TransactionStats{
		Committed:  result.Committed,
		RolledBack: result.RolledBack,
		DeadLocks:  result.Deadlocks,
	}

	// AIDEV-NOTE: 250903170308 Calculate conflict rate
	total := result.Committed + result.RolledBack
	if total > 0 {
		stats.ConflictRate = float64(result.Conflicts) / float64(total)
	}

	return stats, nil
}

// AIDEV-NOTE: 250903170309 Deadlock collector for critical error detection
type DeadlockCollector struct{}

func (c *DeadlockCollector) Name() string {
	return "deadlocks"
}

func (c *DeadlockCollector) Interval() time.Duration {
	return 30 * time.Second
}

func (c *DeadlockCollector) Collect(ctx context.Context, db *sqlx.DB) (interface{}, error) {
	query := `
		SELECT deadlocks
		FROM pg_stat_database 
		WHERE datname = current_database()`

	var deadlocks int64
	err := db.GetContext(ctx, &deadlocks, query)
	if err != nil {
		return int64(0), errors.WrapDatabase(err, "collect_deadlocks")
	}

	return deadlocks, nil
}

// AIDEV-NOTE: 250903170310 Cache hit ratio collector for performance tuning
type CacheHitRatioCollector struct{}

func (c *CacheHitRatioCollector) Name() string {
	return "cache_hit_ratio"
}

func (c *CacheHitRatioCollector) Interval() time.Duration {
	return 2 * time.Minute
}

func (c *CacheHitRatioCollector) Collect(ctx context.Context, db *sqlx.DB) (interface{}, error) {
	query := `
		SELECT 
			sum(heap_blks_hit) as heap_hit,
			sum(heap_blks_read) as heap_read,
			sum(idx_blks_hit) as idx_hit,
			sum(idx_blks_read) as idx_read
		FROM pg_statio_user_tables`

	type cacheResult struct {
		HeapHit  int64 `db:"heap_hit"`
		HeapRead int64 `db:"heap_read"`
		IdxHit   int64 `db:"idx_hit"`
		IdxRead  int64 `db:"idx_read"`
	}

	var result cacheResult
	err := db.GetContext(ctx, &result, query)
	if err != nil {
		return float64(0), errors.WrapDatabase(err, "collect_cache_hit_ratio")
	}

	// AIDEV-NOTE: 250903170311 Calculate overall cache hit ratio
	totalHits := result.HeapHit + result.IdxHit
	totalReads := result.HeapRead + result.IdxRead
	total := totalHits + totalReads

	if total == 0 {
		return float64(1.0), nil // Perfect hit ratio when no reads
	}

	ratio := float64(totalHits) / float64(total)
	return ratio, nil
}

// AIDEV-NOTE: 250903170312 Table bloat collector for maintenance insights
type TableBloatCollector struct{}

func (c *TableBloatCollector) Name() string {
	return "table_bloat"
}

func (c *TableBloatCollector) Interval() time.Duration {
	return 15 * time.Minute
}

func (c *TableBloatCollector) Collect(ctx context.Context, db *sqlx.DB) (interface{}, error) {
	query := `
		SELECT 
			schemaname,
			tablename,
			n_tup_ins as inserts,
			n_tup_upd as updates,
			n_tup_del as deletes,
			n_dead_tup as dead_tuples,
			CASE 
				WHEN n_live_tup > 0 THEN round((n_dead_tup::float / n_live_tup::float) * 100, 2)
				ELSE 0
			END as bloat_ratio
		FROM pg_stat_user_tables 
		WHERE schemaname = 'public'
		ORDER BY n_dead_tup DESC`

	type bloatResult struct {
		SchemaName  string  `db:"schemaname"`
		TableName   string  `db:"tablename"`
		Inserts     int64   `db:"inserts"`
		Updates     int64   `db:"updates"`
		Deletes     int64   `db:"deletes"`
		DeadTuples  int64   `db:"dead_tuples"`
		BloatRatio  float64 `db:"bloat_ratio"`
	}

	var results []bloatResult
	err := db.SelectContext(ctx, &results, query)
	if err != nil {
		return nil, errors.WrapDatabase(err, "collect_table_bloat")
	}

	bloatStats := make(map[string]*TableBloatStat)
	for _, result := range results {
		key := fmt.Sprintf("%s.%s", result.SchemaName, result.TableName)
		bloatStats[key] = &TableBloatStat{
			SchemaName: result.SchemaName,
			TableName:  result.TableName,
			Inserts:    result.Inserts,
			Updates:    result.Updates,
			Deletes:    result.Deletes,
			DeadTuples: result.DeadTuples,
			BloatRatio: result.BloatRatio,
		}
	}

	return bloatStats, nil
}

// AIDEV-NOTE: 250903170313 Lock monitoring collector for contention analysis
type LockMonitorCollector struct{}

func (c *LockMonitorCollector) Name() string {
	return "lock_monitor"
}

func (c *LockMonitorCollector) Interval() time.Duration {
	return 30 * time.Second
}

func (c *LockMonitorCollector) Collect(ctx context.Context, db *sqlx.DB) (interface{}, error) {
	query := `
		SELECT 
			mode,
			count(*) as count,
			avg(extract(epoch from (now() - query_start))) as avg_duration_seconds
		FROM pg_locks l
		JOIN pg_stat_activity a ON l.pid = a.pid
		WHERE l.granted = true
		  AND a.datname = current_database()
		GROUP BY mode
		ORDER BY count DESC`

	type lockResult struct {
		Mode            string  `db:"mode"`
		Count           int     `db:"count"`
		AvgDurationSecs float64 `db:"avg_duration_seconds"`
	}

	var results []lockResult
	err := db.SelectContext(ctx, &results, query)
	if err != nil {
		return nil, errors.WrapDatabase(err, "collect_lock_monitor")
	}

	lockStats := make(map[string]*LockStat)
	for _, result := range results {
		lockStats[result.Mode] = &LockStat{
			Mode:        result.Mode,
			Count:       result.Count,
			AvgDuration: time.Duration(result.AvgDurationSecs * float64(time.Second)),
		}
	}

	return lockStats, nil
}

// AIDEV-NOTE: 250903170314 Replication lag collector for high availability monitoring
type ReplicationLagCollector struct{}

func (c *ReplicationLagCollector) Name() string {
	return "replication_lag"
}

func (c *ReplicationLagCollector) Interval() time.Duration {
	return 1 * time.Minute
}

func (c *ReplicationLagCollector) Collect(ctx context.Context, db *sqlx.DB) (interface{}, error) {
	// AIDEV-NOTE: 250903170315 Check if this is a replica by looking for recovery info
	query := `
		SELECT 
			CASE 
				WHEN pg_is_in_recovery() THEN 
					EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))
				ELSE 0
			END as lag_seconds`

	var lagSeconds float64
	err := db.GetContext(ctx, &lagSeconds, query)
	if err != nil {
		return time.Duration(0), errors.WrapDatabase(err, "collect_replication_lag")
	}

	lag := time.Duration(lagSeconds * float64(time.Second))
	return lag, nil
}

// AIDEV-NOTE: 250903170316 Supporting data structures for collectors

type TableBloatStat struct {
	SchemaName string  `json:"schema_name"`
	TableName  string  `json:"table_name"`
	Inserts    int64   `json:"inserts"`
	Updates    int64   `json:"updates"`
	Deletes    int64   `json:"deletes"`
	DeadTuples int64   `json:"dead_tuples"`
	BloatRatio float64 `json:"bloat_ratio"`
}

type LockStat struct {
	Mode        string        `json:"mode"`
	Count       int           `json:"count"`
	AvgDuration time.Duration `json:"avg_duration"`
}

// AIDEV-NOTE: 250903170317 Query statistics collector for detailed analysis
type QueryStatsCollector struct {
	minCalls int64
}

func NewQueryStatsCollector(minCalls int64) *QueryStatsCollector {
	return &QueryStatsCollector{
		minCalls: minCalls,
	}
}

func (c *QueryStatsCollector) Name() string {
	return "query_stats"
}

func (c *QueryStatsCollector) Interval() time.Duration {
	return 5 * time.Minute
}

func (c *QueryStatsCollector) Collect(ctx context.Context, db *sqlx.DB) (interface{}, error) {
	// AIDEV-NOTE: 250903170318 Comprehensive query statistics
	query := `
		SELECT 
			left(query, 100) as query_sample,
			calls,
			total_time,
			mean_time,
			min_time,
			max_time,
			stddev_time,
			rows,
			100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) as hit_percent
		FROM pg_stat_statements 
		WHERE calls >= $1
		  AND query NOT LIKE '%pg_stat_statements%'
		ORDER BY total_time DESC 
		LIMIT 50`

	type queryStatsResult struct {
		QuerySample string   `db:"query_sample"`
		Calls       int64    `db:"calls"`
		TotalTime   float64  `db:"total_time"`
		MeanTime    float64  `db:"mean_time"`
		MinTime     float64  `db:"min_time"`
		MaxTime     float64  `db:"max_time"`
		StddevTime  float64  `db:"stddev_time"`
		Rows        int64    `db:"rows"`
		HitPercent  *float64 `db:"hit_percent"`
	}

	var results []queryStatsResult
	err := db.SelectContext(ctx, &results, query, c.minCalls)
	if err != nil {
		// Return empty stats if pg_stat_statements is not available
		return make([]*QueryStatDetail, 0), nil
	}

	stats := make([]*QueryStatDetail, 0, len(results))
	for _, result := range results {
		hitPercent := float64(0)
		if result.HitPercent != nil {
			hitPercent = *result.HitPercent
		}

		stats = append(stats, &QueryStatDetail{
			QuerySample: result.QuerySample,
			Calls:       result.Calls,
			TotalTime:   time.Duration(result.TotalTime * float64(time.Millisecond)),
			MeanTime:    time.Duration(result.MeanTime * float64(time.Millisecond)),
			MinTime:     time.Duration(result.MinTime * float64(time.Millisecond)),
			MaxTime:     time.Duration(result.MaxTime * float64(time.Millisecond)),
			StddevTime:  time.Duration(result.StddevTime * float64(time.Millisecond)),
			Rows:        result.Rows,
			HitPercent:  hitPercent,
		})
	}

	return stats, nil
}

type QueryStatDetail struct {
	QuerySample string        `json:"query_sample"`
	Calls       int64         `json:"calls"`
	TotalTime   time.Duration `json:"total_time"`
	MeanTime    time.Duration `json:"mean_time"`
	MinTime     time.Duration `json:"min_time"`
	MaxTime     time.Duration `json:"max_time"`
	StddevTime  time.Duration `json:"stddev_time"`
	Rows        int64         `json:"rows"`
	HitPercent  float64       `json:"hit_percent"`
}