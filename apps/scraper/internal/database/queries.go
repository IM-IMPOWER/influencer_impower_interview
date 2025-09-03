// AIDEV-NOTE: 250903170100 Optimized database queries with prepared statements and caching
// Provides high-performance query execution with monitoring and automatic optimization
package database

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/jmoiron/sqlx"
	"kol-scraper/internal/errors"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: 250903170101 Query executor with prepared statements and caching
type QueryExecutor struct {
	db              *sqlx.DB
	poolManager     *PoolManager
	logger          *logger.Logger
	preparedStmts   map[string]*sqlx.Stmt
	namedStmts      map[string]*sqlx.NamedStmt
	stmtMutex       sync.RWMutex
	queryCache      *QueryCache
	metrics         *QueryMetrics
	slowQueryThreshold time.Duration
}

// AIDEV-NOTE: 250903170102 Query cache for frequently executed queries
type QueryCache struct {
	mu       sync.RWMutex
	cache    map[string]*CacheEntry
	maxSize  int
	ttl      time.Duration
	hits     int64
	misses   int64
}

type CacheEntry struct {
	data      interface{}
	timestamp time.Time
	hits      int64
}

// AIDEV-NOTE: 250903170103 Query execution metrics
type QueryMetrics struct {
	mu                sync.RWMutex
	TotalQueries      int64                    `json:"total_queries"`
	SlowQueries       int64                    `json:"slow_queries"`
	FailedQueries     int64                    `json:"failed_queries"`
	CacheHits         int64                    `json:"cache_hits"`
	CacheMisses       int64                    `json:"cache_misses"`
	QueryLatencies    map[string]time.Duration `json:"query_latencies"`
	PreparedStatements int                     `json:"prepared_statements"`
	LastReset         time.Time                `json:"last_reset"`
}

// AIDEV-NOTE: 250903170104 Commonly used prepared statements
var (
	// KOL queries
	selectKOLByPlatformUserStmt = `
		SELECT id, username, display_name, platform, platform_id, profile_url,
			   avatar_url, bio, location, tier, primary_category, secondary_categories,
			   age_range, gender, languages, is_verified, is_active, is_brand_safe,
			   safety_notes, data_source, last_scraped, scrape_quality_score,
			   content_embedding, created_at, updated_at
		FROM kols 
		WHERE platform = $1 AND username = $2 AND is_active = true`

	selectKOLsByFiltersStmt = `
		SELECT id, username, display_name, platform, platform_id, profile_url,
			   avatar_url, bio, location, tier, primary_category, secondary_categories,
			   age_range, gender, languages, is_verified, is_active, is_brand_safe,
			   safety_notes, data_source, last_scraped, scrape_quality_score,
			   content_embedding, created_at, updated_at
		FROM kols 
		WHERE is_active = true
		  AND ($1::text IS NULL OR platform = $1)
		  AND ($2::text IS NULL OR tier = $2)
		  AND ($3::text IS NULL OR primary_category = $3)
		  AND ($4::boolean IS NULL OR is_verified = $4)
		  AND ($5::boolean IS NULL OR is_brand_safe = $5)
		ORDER BY last_scraped DESC, created_at DESC
		LIMIT $6 OFFSET $7`

	// Metrics queries
	selectLatestMetricsStmt = `
		SELECT id, kol_id, follower_count, following_count, total_posts, total_videos,
			   avg_likes, avg_comments, avg_shares, avg_views, engagement_rate,
			   like_rate, comment_rate, audience_quality_score, fake_follower_percentage,
			   posts_last_30_days, avg_posting_frequency, follower_growth_rate,
			   engagement_trend, metrics_date, created_at, updated_at
		FROM kol_metrics 
		WHERE kol_id = $1 
		ORDER BY metrics_date DESC 
		LIMIT 1`

	selectKOLsWithMetricsStmt = `
		SELECT k.id, k.username, k.display_name, k.platform, k.tier, k.primary_category,
			   k.is_verified, k.is_brand_safe, k.last_scraped,
			   m.follower_count, m.engagement_rate, m.metrics_date
		FROM kols k
		LEFT JOIN LATERAL (
			SELECT follower_count, engagement_rate, metrics_date
			FROM kol_metrics km
			WHERE km.kol_id = k.id
			ORDER BY metrics_date DESC
			LIMIT 1
		) m ON true
		WHERE k.is_active = true
		  AND ($1::text IS NULL OR k.platform = $1)
		  AND ($2::text IS NULL OR k.tier = $2)
		  AND ($3::int IS NULL OR m.follower_count >= $3)
		  AND ($4::int IS NULL OR m.follower_count <= $4)
		  AND ($5::float IS NULL OR m.engagement_rate >= $5)
		ORDER BY m.follower_count DESC NULLS LAST
		LIMIT $6 OFFSET $7`
	
	// Content queries
	selectRecentContentStmt = `
		SELECT id, kol_id, platform_content_id, content_type, content_url,
			   caption, hashtags, mentions, likes_count, comments_count,
			   shares_count, views_count, content_categories, brand_mentions,
			   sentiment_score, content_embedding, posted_at, created_at, updated_at
		FROM kol_content 
		WHERE kol_id = $1 AND posted_at >= $2
		ORDER BY posted_at DESC
		LIMIT $3`
)

// NewQueryExecutor creates a new optimized query executor
func NewQueryExecutor(db *sqlx.DB, poolManager *PoolManager, logger *logger.Logger) *QueryExecutor {
	qe := &QueryExecutor{
		db:                 db,
		poolManager:        poolManager,
		logger:             logger,
		preparedStmts:      make(map[string]*sqlx.Stmt),
		namedStmts:         make(map[string]*sqlx.NamedStmt),
		slowQueryThreshold: 1 * time.Second,
		queryCache: &QueryCache{
			cache:   make(map[string]*CacheEntry),
			maxSize: 1000,
			ttl:     5 * time.Minute,
		},
		metrics: &QueryMetrics{
			QueryLatencies: make(map[string]time.Duration),
			LastReset:      time.Now(),
		},
	}

	// AIDEV-NOTE: 250903170105 Prepare frequently used statements
	qe.prepareCommonStatements()

	return qe
}

// prepareCommonStatements prepares frequently used SQL statements
func (qe *QueryExecutor) prepareCommonStatements() {
	statements := map[string]string{
		"selectKOLByPlatformUser": selectKOLByPlatformUserStmt,
		"selectKOLsByFilters":     selectKOLsByFiltersStmt,
		"selectLatestMetrics":     selectLatestMetricsStmt,
		"selectKOLsWithMetrics":   selectKOLsWithMetricsStmt,
		"selectRecentContent":     selectRecentContentStmt,
	}

	qe.stmtMutex.Lock()
	defer qe.stmtMutex.Unlock()

	for name, query := range statements {
		if stmt, err := qe.db.Preparex(query); err != nil {
			qe.logger.ErrorLog(errors.WrapDatabase(err, fmt.Sprintf("prepare_%s", name)), 
				"failed_to_prepare_statement", logger.Fields{
					"statement_name": name,
					"query":          query,
				})
		} else {
			qe.preparedStmts[name] = stmt
			qe.logger.Debug("Prepared statement created", logger.Fields{
				"statement_name": name,
			})
		}
	}

	qe.metrics.mu.Lock()
	qe.metrics.PreparedStatements = len(qe.preparedStmts)
	qe.metrics.mu.Unlock()

	qe.logger.Info("Database prepared statements initialized", logger.Fields{
		"total_prepared": len(qe.preparedStmts),
	})
}

// AIDEV-NOTE: 250903170106 Optimized KOL retrieval methods
func (qe *QueryExecutor) GetKOLByPlatformAndUsername(ctx context.Context, platform, username string) (*KOL, error) {
	start := time.Now()
	defer qe.recordQueryMetric("GetKOLByPlatformAndUsername", start)

	// AIDEV-NOTE: 250903170107 Try cache first
	cacheKey := fmt.Sprintf("kol:%s:%s", platform, username)
	if cached := qe.queryCache.get(cacheKey); cached != nil {
		if kol, ok := cached.(*KOL); ok {
			qe.metrics.mu.Lock()
			qe.metrics.CacheHits++
			qe.metrics.mu.Unlock()
			return kol, nil
		}
	}

	qe.stmtMutex.RLock()
	stmt, exists := qe.preparedStmts["selectKOLByPlatformUser"]
	qe.stmtMutex.RUnlock()

	if !exists {
		return nil, errors.New(errors.CategoryDatabase, errors.CodeDatabaseQuery,
			"Prepared statement not found: selectKOLByPlatformUser").Build()
	}

	var kol KOL
	err := stmt.GetContext(ctx, &kol, platform, username)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, errors.NotFoundError("KOL")
		}
		return nil, errors.WrapDatabase(err, "GetKOLByPlatformAndUsername")
	}

	// AIDEV-NOTE: 250903170108 Cache the result
	qe.queryCache.set(cacheKey, &kol)
	qe.metrics.mu.Lock()
	qe.metrics.CacheMisses++
	qe.metrics.mu.Unlock()

	return &kol, nil
}

// GetKOLsByFilters retrieves KOLs with specified filters using prepared statements
func (qe *QueryExecutor) GetKOLsByFilters(ctx context.Context, filters *KOLFilters) ([]*KOL, error) {
	start := time.Now()
	defer qe.recordQueryMetric("GetKOLsByFilters", start)

	qe.stmtMutex.RLock()
	stmt, exists := qe.preparedStmts["selectKOLsByFilters"]
	qe.stmtMutex.RUnlock()

	if !exists {
		return nil, errors.New(errors.CategoryDatabase, errors.CodeDatabaseQuery,
			"Prepared statement not found: selectKOLsByFilters").Build()
	}

	// AIDEV-NOTE: 250903170109 Build cache key from filters
	cacheKey := qe.buildFiltersKey(filters)
	if cached := qe.queryCache.get(cacheKey); cached != nil {
		if kols, ok := cached.([]*KOL); ok {
			qe.metrics.mu.Lock()
			qe.metrics.CacheHits++
			qe.metrics.mu.Unlock()
			return kols, nil
		}
	}

	var kols []*KOL
	err := stmt.SelectContext(ctx, &kols,
		nullString(filters.Platform),
		nullString(filters.Tier),
		nullString(filters.PrimaryCategory),
		nullBool(filters.IsVerified),
		nullBool(filters.IsBrandSafe),
		filters.Limit,
		filters.Offset,
	)

	if err != nil {
		return nil, errors.WrapDatabase(err, "GetKOLsByFilters").
			WithContext("filters", filters)
	}

	// AIDEV-NOTE: 250903170110 Cache the results
	qe.queryCache.set(cacheKey, kols)
	qe.metrics.mu.Lock()
	qe.metrics.CacheMisses++
	qe.metrics.mu.Unlock()

	return kols, nil
}

// GetKOLsWithMetrics retrieves KOLs with their latest metrics efficiently
func (qe *QueryExecutor) GetKOLsWithMetrics(ctx context.Context, filters *KOLMetricsFilters) ([]*KOLWithMetrics, error) {
	start := time.Now()
	defer qe.recordQueryMetric("GetKOLsWithMetrics", start)

	qe.stmtMutex.RLock()
	stmt, exists := qe.preparedStmts["selectKOLsWithMetrics"]
	qe.stmtMutex.RUnlock()

	if !exists {
		return nil, errors.New(errors.CategoryDatabase, errors.CodeDatabaseQuery,
			"Prepared statement not found: selectKOLsWithMetrics").Build()
	}

	var results []*KOLWithMetrics
	err := stmt.SelectContext(ctx, &results,
		nullString(filters.Platform),
		nullString(filters.Tier),
		nullInt(filters.MinFollowers),
		nullInt(filters.MaxFollowers),
		nullFloat64(filters.MinEngagementRate),
		filters.Limit,
		filters.Offset,
	)

	if err != nil {
		return nil, errors.WrapDatabase(err, "GetKOLsWithMetrics").
			WithContext("filters", filters)
	}

	return results, nil
}

// GetLatestMetrics retrieves the latest metrics for a KOL using prepared statements
func (qe *QueryExecutor) GetLatestMetrics(ctx context.Context, kolID string) (*KOLMetrics, error) {
	start := time.Now()
	defer qe.recordQueryMetric("GetLatestMetrics", start)

	// AIDEV-NOTE: 250903170111 Check cache first
	cacheKey := fmt.Sprintf("metrics:%s", kolID)
	if cached := qe.queryCache.get(cacheKey); cached != nil {
		if metrics, ok := cached.(*KOLMetrics); ok {
			qe.metrics.mu.Lock()
			qe.metrics.CacheHits++
			qe.metrics.mu.Unlock()
			return metrics, nil
		}
	}

	qe.stmtMutex.RLock()
	stmt, exists := qe.preparedStmts["selectLatestMetrics"]
	qe.stmtMutex.RUnlock()

	if !exists {
		return nil, errors.New(errors.CategoryDatabase, errors.CodeDatabaseQuery,
			"Prepared statement not found: selectLatestMetrics").Build()
	}

	var metrics KOLMetrics
	err := stmt.GetContext(ctx, &metrics, kolID)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, errors.NotFoundError("KOL metrics")
		}
		return nil, errors.WrapDatabase(err, "GetLatestMetrics").
			WithContext("kol_id", kolID)
	}

	// AIDEV-NOTE: 250903170112 Cache with shorter TTL for metrics
	qe.queryCache.setWithTTL(cacheKey, &metrics, 2*time.Minute)
	qe.metrics.mu.Lock()
	qe.metrics.CacheMisses++
	qe.metrics.mu.Unlock()

	return &metrics, nil
}

// GetRecentContent retrieves recent content for a KOL
func (qe *QueryExecutor) GetRecentContent(ctx context.Context, kolID string, since time.Time, limit int) ([]*KOLContent, error) {
	start := time.Now()
	defer qe.recordQueryMetric("GetRecentContent", start)

	qe.stmtMutex.RLock()
	stmt, exists := qe.preparedStmts["selectRecentContent"]
	qe.stmtMutex.RUnlock()

	if !exists {
		return nil, errors.New(errors.CategoryDatabase, errors.CodeDatabaseQuery,
			"Prepared statement not found: selectRecentContent").Build()
	}

	var content []*KOLContent
	err := stmt.SelectContext(ctx, &content, kolID, since, limit)
	if err != nil {
		return nil, errors.WrapDatabase(err, "GetRecentContent").
			WithContext("kol_id", kolID).
			WithContext("since", since)
	}

	return content, nil
}

// AIDEV-NOTE: 250903170113 Batch operations for high-performance inserts
func (qe *QueryExecutor) BatchUpsertKOLs(ctx context.Context, kols []*KOL, batchSize int) error {
	if len(kols) == 0 {
		return nil
	}

	start := time.Now()
	defer qe.recordQueryMetric("BatchUpsertKOLs", start)

	// AIDEV-NOTE: 250903170114 Process in optimized batches
	return qe.processBatches(ctx, kols, batchSize, qe.batchUpsertKOLsInternal)
}

// batchUpsertKOLsInternal performs the actual batch upsert
func (qe *QueryExecutor) batchUpsertKOLsInternal(ctx context.Context, batch []*KOL) error {
	tx, err := qe.db.BeginTxx(ctx, nil)
	if err != nil {
		return errors.WrapDatabase(err, "begin_transaction")
	}
	defer tx.Rollback()

	// AIDEV-NOTE: 250903170115 Use COPY for optimal performance on large batches
	if len(batch) > 50 {
		return qe.copyKOLs(ctx, tx, batch)
	}

	// AIDEV-NOTE: 250903170116 Use regular upsert for smaller batches
	stmt, err := tx.PrepareNamedContext(ctx, `
		INSERT INTO kols (
			id, username, display_name, platform, platform_id, profile_url,
			avatar_url, bio, location, tier, primary_category, secondary_categories,
			age_range, gender, languages, is_verified, is_active, is_brand_safe,
			safety_notes, data_source, last_scraped, scrape_quality_score,
			content_embedding, created_at, updated_at
		) VALUES (
			:id, :username, :display_name, :platform, :platform_id, :profile_url,
			:avatar_url, :bio, :location, :tier, :primary_category, :secondary_categories,
			:age_range, :gender, :languages, :is_verified, :is_active, :is_brand_safe,
			:safety_notes, :data_source, :last_scraped, :scrape_quality_score,
			:content_embedding, :created_at, :updated_at
		) ON CONFLICT (platform, platform_id) 
		DO UPDATE SET
			username = EXCLUDED.username,
			display_name = EXCLUDED.display_name,
			profile_url = EXCLUDED.profile_url,
			avatar_url = EXCLUDED.avatar_url,
			bio = EXCLUDED.bio,
			location = EXCLUDED.location,
			tier = EXCLUDED.tier,
			primary_category = EXCLUDED.primary_category,
			secondary_categories = EXCLUDED.secondary_categories,
			age_range = EXCLUDED.age_range,
			gender = EXCLUDED.gender,
			languages = EXCLUDED.languages,
			is_verified = EXCLUDED.is_verified,
			is_active = EXCLUDED.is_active,
			is_brand_safe = EXCLUDED.is_brand_safe,
			safety_notes = EXCLUDED.safety_notes,
			data_source = EXCLUDED.data_source,
			last_scraped = EXCLUDED.last_scraped,
			scrape_quality_score = EXCLUDED.scrape_quality_score,
			content_embedding = EXCLUDED.content_embedding,
			updated_at = EXCLUDED.updated_at`)

	if err != nil {
		return errors.WrapDatabase(err, "prepare_upsert_statement")
	}
	defer stmt.Close()

	for _, kol := range batch {
		if _, err := stmt.ExecContext(ctx, kol); err != nil {
			return errors.WrapDatabase(err, "execute_kol_upsert").
				WithContext("kol_id", kol.ID)
		}
	}

	return tx.Commit()
}

// AIDEV-NOTE: 250903170117 Cache management methods
func (qc *QueryCache) get(key string) interface{} {
	qc.mu.RLock()
	defer qc.mu.RUnlock()

	entry, exists := qc.cache[key]
	if !exists {
		return nil
	}

	// Check if entry has expired
	if time.Since(entry.timestamp) > qc.ttl {
		delete(qc.cache, key)
		return nil
	}

	entry.hits++
	return entry.data
}

func (qc *QueryCache) set(key string, value interface{}) {
	qc.setWithTTL(key, value, qc.ttl)
}

func (qc *QueryCache) setWithTTL(key string, value interface{}, ttl time.Duration) {
	qc.mu.Lock()
	defer qc.mu.Unlock()

	// AIDEV-NOTE: 250903170118 Implement simple LRU eviction
	if len(qc.cache) >= qc.maxSize {
		qc.evictOldest()
	}

	qc.cache[key] = &CacheEntry{
		data:      value,
		timestamp: time.Now(),
		hits:      0,
	}
}

func (qc *QueryCache) evictOldest() {
	var oldestKey string
	var oldestTime time.Time

	for key, entry := range qc.cache {
		if oldestKey == "" || entry.timestamp.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.timestamp
		}
	}

	if oldestKey != "" {
		delete(qc.cache, oldestKey)
	}
}

// AIDEV-NOTE: 250903170119 Utility functions for database operations
func (qe *QueryExecutor) recordQueryMetric(operation string, start time.Time) {
	duration := time.Since(start)

	qe.metrics.mu.Lock()
	defer qe.metrics.mu.Unlock()

	qe.metrics.TotalQueries++
	qe.metrics.QueryLatencies[operation] = duration

	if duration > qe.slowQueryThreshold {
		qe.metrics.SlowQueries++
		qe.logger.Warn("Slow query detected", logger.Fields{
			"operation":    operation,
			"duration_ms":  duration.Milliseconds(),
			"threshold_ms": qe.slowQueryThreshold.Milliseconds(),
		})
	}
}

func (qe *QueryExecutor) processBatches(ctx context.Context, items interface{}, batchSize int, processor func(context.Context, interface{}) error) error {
	// This would be implemented based on the specific batch type
	// For now, return a placeholder
	return fmt.Errorf("batch processing not implemented for this type")
}

func (qe *QueryExecutor) copyKOLs(ctx context.Context, tx *sqlx.Tx, kols []*KOL) error {
	// AIDEV-NOTE: 250903170120 Implement COPY for high-performance bulk inserts
	// This would use PostgreSQL's COPY protocol for maximum performance
	return fmt.Errorf("COPY implementation pending")
}

func (qe *QueryExecutor) buildFiltersKey(filters *KOLFilters) string {
	return fmt.Sprintf("filters:%s:%s:%s:%v:%v:%d:%d",
		nullString(filters.Platform),
		nullString(filters.Tier),
		nullString(filters.PrimaryCategory),
		filters.IsVerified,
		filters.IsBrandSafe,
		filters.Limit,
		filters.Offset,
	)
}

// GetQueryMetrics returns current query execution metrics
func (qe *QueryExecutor) GetQueryMetrics() *QueryMetrics {
	qe.metrics.mu.RLock()
	defer qe.metrics.mu.RUnlock()

	// Create a copy to avoid race conditions
	latencies := make(map[string]time.Duration)
	for k, v := range qe.metrics.QueryLatencies {
		latencies[k] = v
	}

	return &QueryMetrics{
		TotalQueries:       qe.metrics.TotalQueries,
		SlowQueries:        qe.metrics.SlowQueries,
		FailedQueries:      qe.metrics.FailedQueries,
		CacheHits:          qe.metrics.CacheHits,
		CacheMisses:        qe.metrics.CacheMisses,
		QueryLatencies:     latencies,
		PreparedStatements: qe.metrics.PreparedStatements,
		LastReset:          qe.metrics.LastReset,
	}
}

// InvalidateCache clears the query cache
func (qe *QueryExecutor) InvalidateCache() {
	qe.queryCache.mu.Lock()
	defer qe.queryCache.mu.Unlock()

	qe.queryCache.cache = make(map[string]*CacheEntry)
	qe.logger.Info("Query cache invalidated")
}

// Close closes prepared statements and cleans up resources
func (qe *QueryExecutor) Close() error {
	qe.stmtMutex.Lock()
	defer qe.stmtMutex.Unlock()

	// AIDEV-NOTE: 250903170121 Close all prepared statements
	for name, stmt := range qe.preparedStmts {
		if err := stmt.Close(); err != nil {
			qe.logger.ErrorLog(errors.WrapDatabase(err, "close_prepared_statement"),
				"failed_to_close_statement", logger.Fields{"statement_name": name})
		}
	}

	for name, stmt := range qe.namedStmts {
		if err := stmt.Close(); err != nil {
			qe.logger.ErrorLog(errors.WrapDatabase(err, "close_named_statement"),
				"failed_to_close_statement", logger.Fields{"statement_name": name})
		}
	}

	qe.logger.Info("Query executor closed")
	return nil
}

// AIDEV-NOTE: 250903170122 Helper functions for null handling
func nullString(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}

func nullBool(b *bool) interface{} {
	if b == nil {
		return nil
	}
	return *b
}

func nullInt(i *int) interface{} {
	if i == nil {
		return nil
	}
	return *i
}

func nullFloat64(f *float64) interface{} {
	if f == nil {
		return nil
	}
	return *f
}

// AIDEV-NOTE: 250903170123 Filter and result structures
type KOLFilters struct {
	Platform        string `json:"platform,omitempty"`
	Tier            string `json:"tier,omitempty"`
	PrimaryCategory string `json:"primary_category,omitempty"`
	IsVerified      *bool  `json:"is_verified,omitempty"`
	IsBrandSafe     *bool  `json:"is_brand_safe,omitempty"`
	Limit           int    `json:"limit"`
	Offset          int    `json:"offset"`
}

type KOLMetricsFilters struct {
	Platform           string   `json:"platform,omitempty"`
	Tier               string   `json:"tier,omitempty"`
	MinFollowers       *int     `json:"min_followers,omitempty"`
	MaxFollowers       *int     `json:"max_followers,omitempty"`
	MinEngagementRate  *float64 `json:"min_engagement_rate,omitempty"`
	Limit              int      `json:"limit"`
	Offset             int      `json:"offset"`
}

type KOLWithMetrics struct {
	KOL
	FollowerCount  int64     `db:"follower_count" json:"follower_count"`
	EngagementRate float64   `db:"engagement_rate" json:"engagement_rate"`
	MetricsDate    time.Time `db:"metrics_date" json:"metrics_date"`
}