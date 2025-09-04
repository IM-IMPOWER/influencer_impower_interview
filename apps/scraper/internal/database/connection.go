// AIDEV-NOTE: 250903170500 Enhanced database connection management for KOL scraper service
// Handles PostgreSQL connections with advanced pooling, monitoring, and resilience features
package database

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	_ "github.com/golang-migrate/migrate/v4/source/file"
	"github.com/jmoiron/sqlx"
	_ "github.com/lib/pq"
	"kol-scraper/internal/errors"
	"kol-scraper/pkg/config"
	"kol-scraper/pkg/logger"
)

// AIDEV-NOTE: 250903170501 Enhanced DB wrapper with optimization components
type DB struct {
	*sqlx.DB
	poolManager     *PoolManager
	queryExecutor   *QueryExecutor
	monitor         *DatabaseMonitor
	resilientClient *ResilientDB
	logger          *logger.Logger
	config          *config.Config
}

// NewConnection creates a new optimized database connection with all enhancement components
func NewConnection(cfg *config.Config, log *logger.Logger) (*DB, error) {
	// AIDEV-NOTE: 250903170502 Connect to PostgreSQL database
	db, err := sqlx.Connect("postgres", cfg.DatabaseURL)
	if err != nil {
		return nil, errors.WrapDatabase(err, "connect_to_database")
	}

	log.Info("Database connection established", logger.Fields{
		"max_connections":    cfg.MaxConnections,
		"max_idle_conns":     cfg.MaxIdleConns,
		"conn_max_lifetime":  cfg.ConnMaxLifetime,
	})

	// AIDEV-NOTE: 250903170503 Initialize pool manager for advanced connection management
	poolManager := NewPoolManager(db, cfg, log)

	// AIDEV-NOTE: 250903170504 Test connection with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		return nil, errors.WrapDatabase(err, "ping_database")
	}

	// AIDEV-NOTE: 250903170505 Enable required PostgreSQL extensions
	if err := enableExtensions(db, log); err != nil {
		return nil, errors.WrapDatabase(err, "enable_extensions")
	}

	// AIDEV-NOTE: 250903170506 Initialize query executor with prepared statements
	queryExecutor := NewQueryExecutor(db, poolManager, log)

	// AIDEV-NOTE: 250903170507 Initialize database monitor for production visibility
	monitor := NewDatabaseMonitor(db, poolManager, log)

	// AIDEV-NOTE: 250903170508 Initialize resilient client for retry capabilities
	resilientClient := NewResilientDB(db, poolManager, log)

	// AIDEV-NOTE: 250903170509 Start monitoring in background
	go func() {
		if err := monitor.Start(context.Background()); err != nil {
			log.ErrorLog(errors.WrapDatabase(err, "start_database_monitor"),
				"database_monitor_start_failed", nil)
		}
	}()

	enhancedDB := &DB{
		DB:              db,
		poolManager:     poolManager,
		queryExecutor:   queryExecutor,
		monitor:         monitor,
		resilientClient: resilientClient,
		logger:          log,
		config:          cfg,
	}

	log.Info("Enhanced database connection initialized with all optimization components")

	return enhancedDB, nil
}

// enableExtensions enables required PostgreSQL extensions with enhanced error handling
func enableExtensions(db *sqlx.DB, log *logger.Logger) error {
	extensions := []struct {
		name string
		sql  string
	}{
		{"uuid-ossp", "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""},
		{"pg_trgm", "CREATE EXTENSION IF NOT EXISTS \"pg_trgm\""},
		{"vector", "CREATE EXTENSION IF NOT EXISTS \"vector\""},
		{"btree_gin", "CREATE EXTENSION IF NOT EXISTS \"btree_gin\""},
		{"pg_stat_statements", "CREATE EXTENSION IF NOT EXISTS \"pg_stat_statements\""},
	}

	for _, ext := range extensions {
		if _, err := db.Exec(ext.sql); err != nil {
			// AIDEV-NOTE: 250903170510 Some extensions might not be available, log but continue
			log.Warn("Failed to create extension", logger.Fields{
				"extension": ext.name,
				"error":     err.Error(),
			})
			continue
		}
		log.Debug("Extension enabled", logger.Fields{"extension": ext.name})
	}

	return nil
}

// AIDEV-NOTE: 250903170511 Enhanced database operations with optimization components

// Close gracefully closes the database connection and all components
func (db *DB) Close() error {
	db.logger.Info("Closing enhanced database connection")

	// AIDEV-NOTE: 250903170512 Close components in reverse order
	if db.monitor != nil {
		if err := db.monitor.Stop(); err != nil {
			db.logger.ErrorLog(errors.WrapDatabase(err, "close_monitor"), 
				"failed_to_close_monitor", nil)
		}
	}

	if db.resilientClient != nil {
		if err := db.resilientClient.Close(); err != nil {
			db.logger.ErrorLog(errors.WrapDatabase(err, "close_resilient_client"),
				"failed_to_close_resilient_client", nil)
		}
	}

	if db.queryExecutor != nil {
		if err := db.queryExecutor.Close(); err != nil {
			db.logger.ErrorLog(errors.WrapDatabase(err, "close_query_executor"),
				"failed_to_close_query_executor", nil)
		}
	}

	if db.poolManager != nil {
		if err := db.poolManager.Close(); err != nil {
			db.logger.ErrorLog(errors.WrapDatabase(err, "close_pool_manager"),
				"failed_to_close_pool_manager", nil)
		}
	}

	// AIDEV-NOTE: 250903170513 Close the underlying database connection
	return db.DB.Close()
}

// Health checks database connectivity with enhanced monitoring
func (db *DB) Health(ctx context.Context) error {
	if db.resilientClient != nil {
		return db.resilientClient.HealthCheck(ctx)
	}
	return db.PingContext(ctx)
}

// AIDEV-NOTE: 250903170514 Enhanced query methods using optimization components

// QueryWithRetry executes a query using the resilient client
func (db *DB) QueryWithRetry(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	if db.resilientClient != nil {
		return db.resilientClient.QueryContext(ctx, query, args...)
	}
	return db.QueryContext(ctx, query, args...)
}

// ExecWithRetry executes a command using the resilient client
func (db *DB) ExecWithRetry(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	if db.resilientClient != nil {
		return db.resilientClient.ExecContext(ctx, query, args...)
	}
	return db.ExecContext(ctx, query, args...)
}

// GetWithRetry performs a single row query using the resilient client
func (db *DB) GetWithRetry(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	if db.resilientClient != nil {
		return db.resilientClient.GetContext(ctx, dest, query, args...)
	}
	return db.GetContext(ctx, dest, query, args...)
}

// SelectWithRetry performs a multi-row query using the resilient client
func (db *DB) SelectWithRetry(ctx context.Context, dest interface{}, query string, args ...interface{}) error {
	if db.resilientClient != nil {
		return db.resilientClient.SelectContext(ctx, dest, query, args...)
	}
	return db.SelectContext(ctx, dest, query, args...)
}

// TransactionWithRetry executes a transaction using the resilient client
func (db *DB) TransactionWithRetry(ctx context.Context, fn func(*sqlx.Tx) error) error {
	if db.resilientClient != nil {
		return db.resilientClient.TransactionWithRetry(ctx, fn)
	}
	return db.Transaction(ctx, fn)
}

// AIDEV-NOTE: 250903170515 Optimized query methods using prepared statements

// GetKOLByPlatformAndUsername uses optimized prepared statements
func (db *DB) GetKOLByPlatformAndUsername(ctx context.Context, platform, username string) (*KOL, error) {
	if db.queryExecutor != nil {
		return db.queryExecutor.GetKOLByPlatformAndUsername(ctx, platform, username)
	}
	
	// Fallback to direct query
	var kol KOL
	query := `SELECT id, username, display_name, platform, platform_id, profile_url,
			 avatar_url, bio, location, tier, primary_category, secondary_categories,
			 age_range, gender, languages, is_verified, is_active, is_brand_safe,
			 safety_notes, data_source, last_scraped, scrape_quality_score,
			 content_embedding, created_at, updated_at
		FROM kols WHERE platform = $1 AND username = $2 AND is_active = true`
	
	err := db.GetContext(ctx, &kol, query, platform, username)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, errors.NotFoundError("KOL")
		}
		return nil, errors.WrapDatabase(err, "get_kol_by_platform_username")
	}
	return &kol, nil
}

// GetLatestMetrics retrieves latest metrics using optimized queries
func (db *DB) GetLatestMetrics(ctx context.Context, kolID string) (*KOLMetrics, error) {
	if db.queryExecutor != nil {
		return db.queryExecutor.GetLatestMetrics(ctx, kolID)
	}
	
	// Fallback to direct query
	var metrics KOLMetrics
	query := `SELECT id, kol_id, follower_count, following_count, total_posts, total_videos,
			 avg_likes, avg_comments, avg_shares, avg_views, engagement_rate,
			 like_rate, comment_rate, audience_quality_score, fake_follower_percentage,
			 posts_last_30_days, avg_posting_frequency, follower_growth_rate,
			 engagement_trend, metrics_date, created_at, updated_at
		FROM kol_metrics WHERE kol_id = $1 ORDER BY metrics_date DESC LIMIT 1`
	
	err := db.GetContext(ctx, &metrics, query, kolID)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, errors.NotFoundError("KOL metrics")
		}
		return nil, errors.WrapDatabase(err, "get_latest_metrics")
	}
	return &metrics, nil
}

// AIDEV-NOTE: 250903170516 Monitoring and metrics access methods

// GetPoolMetrics returns connection pool metrics
func (db *DB) GetPoolMetrics() *PoolMetrics {
	if db.poolManager != nil {
		return db.poolManager.GetMetrics()
	}
	return nil
}

// GetQueryMetrics returns query execution metrics
func (db *DB) GetQueryMetrics() *QueryMetrics {
	if db.queryExecutor != nil {
		return db.queryExecutor.GetQueryMetrics()
	}
	return nil
}

// GetRetryMetrics returns retry operation metrics
func (db *DB) GetRetryMetrics() *RetryMetrics {
	if db.resilientClient != nil {
		return db.resilientClient.GetRetryMetrics()
	}
	return nil
}

// GetMonitoringMetrics returns database monitoring metrics
func (db *DB) GetMonitoringMetrics() *MonitorMetrics {
	if db.monitor != nil {
		return db.monitor.GetMetrics()
	}
	return nil
}

// GetActiveAlerts returns current database alerts
func (db *DB) GetActiveAlerts() []*ActiveAlert {
	if db.monitor != nil {
		return db.monitor.GetAlerts()
	}
	return nil
}

// GetHealthStatus returns comprehensive health status
func (db *DB) GetHealthStatus() map[string]interface{} {
	status := map[string]interface{}{
		"database": "healthy",
		"timestamp": time.Now(),
	}

	if db.poolManager != nil {
		status["pool"] = db.poolManager.GetHealthStatus()
		status["healthy"] = db.poolManager.IsHealthy()
	}

	if db.resilientClient != nil {
		status["success_rate"] = db.resilientClient.GetSuccessRate()
		status["retry_rate"] = db.resilientClient.GetRetryRate()
	}

	if db.monitor != nil {
		alerts := db.monitor.GetAlerts()
		status["active_alerts"] = len(alerts)
		if len(alerts) > 0 {
			status["healthy"] = false
		}
	}

	return status
}

// InvalidateCache clears query caches
func (db *DB) InvalidateCache() {
	if db.queryExecutor != nil {
		db.queryExecutor.InvalidateCache()
	}
	db.logger.Info("Database caches invalidated")
}

// Migrate runs database migrations
func Migrate(db *DB, migrationsPath string) error {
	// AIDEV-NOTE: Create migrate instance
	driver, err := postgres.WithInstance(db.DB.DB, &postgres.Config{})
	if err != nil {
		return fmt.Errorf("failed to create migrate driver: %w", err)
	}

	m, err := migrate.NewWithDatabaseInstance(
		fmt.Sprintf("file://%s", migrationsPath),
		"postgres",
		driver,
	)
	if err != nil {
		return fmt.Errorf("failed to create migrate instance: %w", err)
	}

	// AIDEV-NOTE: Run migrations
	if err := m.Up(); err != nil && err != migrate.ErrNoChange {
		return fmt.Errorf("failed to run migrations: %w", err)
	}

	return nil
}

// Transaction executes a function within a database transaction
func (db *DB) Transaction(ctx context.Context, fn func(*sqlx.Tx) error) error {
	tx, err := db.BeginTxx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	defer func() {
		if p := recover(); p != nil {
			_ = tx.Rollback()
			panic(p)
		} else if err != nil {
			_ = tx.Rollback()
		} else {
			err = tx.Commit()
		}
	}()

	err = fn(tx)
	return err
}

// BulkInsert performs optimized bulk insert operations
func (db *DB) BulkInsert(ctx context.Context, query string, args []interface{}, batchSize int) error {
	if len(args) == 0 {
		return nil
	}

	// AIDEV-NOTE: Process in batches for memory efficiency
	for i := 0; i < len(args); i += batchSize {
		end := i + batchSize
		if end > len(args) {
			end = len(args)
		}

		batch := args[i:end]
		if _, err := db.ExecContext(ctx, query, batch...); err != nil {
			return fmt.Errorf("bulk insert failed at batch %d: %w", i/batchSize, err)
		}
	}

	return nil
}

// GetStats returns database statistics
func (db *DB) GetStats() sql.DBStats {
	return db.DB.Stats()
}

// AIDEV-NOTE: Database helper functions for common operations

// UpsertKOL inserts or updates a KOL record
func (db *DB) UpsertKOL(ctx context.Context, kol *KOL) error {
	query := `
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
		)
		ON CONFLICT (platform, platform_id) 
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
			updated_at = EXCLUDED.updated_at
		RETURNING id`

	_, err := db.NamedExecContext(ctx, query, kol)
	return err
}

// UpsertKOLMetrics inserts or updates KOL metrics
func (db *DB) UpsertKOLMetrics(ctx context.Context, metrics *KOLMetrics) error {
	query := `
		INSERT INTO kol_metrics (
			id, kol_id, follower_count, following_count, total_posts, total_videos,
			avg_likes, avg_comments, avg_shares, avg_views, engagement_rate,
			like_rate, comment_rate, audience_quality_score, fake_follower_percentage,
			posts_last_30_days, avg_posting_frequency, follower_growth_rate,
			engagement_trend, metrics_date, created_at, updated_at
		) VALUES (
			:id, :kol_id, :follower_count, :following_count, :total_posts, :total_videos,
			:avg_likes, :avg_comments, :avg_shares, :avg_views, :engagement_rate,
			:like_rate, :comment_rate, :audience_quality_score, :fake_follower_percentage,
			:posts_last_30_days, :avg_posting_frequency, :follower_growth_rate,
			:engagement_trend, :metrics_date, :created_at, :updated_at
		)
		ON CONFLICT (kol_id, metrics_date) 
		DO UPDATE SET
			follower_count = EXCLUDED.follower_count,
			following_count = EXCLUDED.following_count,
			total_posts = EXCLUDED.total_posts,
			total_videos = EXCLUDED.total_videos,
			avg_likes = EXCLUDED.avg_likes,
			avg_comments = EXCLUDED.avg_comments,
			avg_shares = EXCLUDED.avg_shares,
			avg_views = EXCLUDED.avg_views,
			engagement_rate = EXCLUDED.engagement_rate,
			like_rate = EXCLUDED.like_rate,
			comment_rate = EXCLUDED.comment_rate,
			audience_quality_score = EXCLUDED.audience_quality_score,
			fake_follower_percentage = EXCLUDED.fake_follower_percentage,
			posts_last_30_days = EXCLUDED.posts_last_30_days,
			avg_posting_frequency = EXCLUDED.avg_posting_frequency,
			follower_growth_rate = EXCLUDED.follower_growth_rate,
			engagement_trend = EXCLUDED.engagement_trend,
			updated_at = EXCLUDED.updated_at`

	_, err := db.NamedExecContext(ctx, query, metrics)
	return err
}

// BulkUpsertKOLContent inserts or updates multiple KOL content records
func (db *DB) BulkUpsertKOLContent(ctx context.Context, content []*KOLContent) error {
	if len(content) == 0 {
		return nil
	}

	return db.Transaction(ctx, func(tx *sqlx.Tx) error {
		query := `
			INSERT INTO kol_content (
				id, kol_id, platform_content_id, content_type, content_url,
				caption, hashtags, mentions, likes_count, comments_count,
				shares_count, views_count, content_categories, brand_mentions,
				sentiment_score, content_embedding, posted_at, created_at, updated_at
			) VALUES (
				:id, :kol_id, :platform_content_id, :content_type, :content_url,
				:caption, :hashtags, :mentions, :likes_count, :comments_count,
				:shares_count, :views_count, :content_categories, :brand_mentions,
				:sentiment_score, :content_embedding, :posted_at, :created_at, :updated_at
			)
			ON CONFLICT (kol_id, platform_content_id) 
			DO UPDATE SET
				content_type = EXCLUDED.content_type,
				content_url = EXCLUDED.content_url,
				caption = EXCLUDED.caption,
				hashtags = EXCLUDED.hashtags,
				mentions = EXCLUDED.mentions,
				likes_count = EXCLUDED.likes_count,
				comments_count = EXCLUDED.comments_count,
				shares_count = EXCLUDED.shares_count,
				views_count = EXCLUDED.views_count,
				content_categories = EXCLUDED.content_categories,
				brand_mentions = EXCLUDED.brand_mentions,
				sentiment_score = EXCLUDED.sentiment_score,
				content_embedding = EXCLUDED.content_embedding,
				updated_at = EXCLUDED.updated_at`

		for _, item := range content {
			if _, err := tx.NamedExecContext(ctx, query, item); err != nil {
				return fmt.Errorf("failed to upsert content %s: %w", item.ID, err)
			}
		}

		return nil
	})
}