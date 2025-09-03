-- AIDEV-NOTE: Database performance optimizations for KOL platform
-- Essential indexes and performance tuning for high-throughput operations

-- Core indexes for KOL queries (POC1 performance critical)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kols_platform_tier 
ON kols (platform, tier) WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kols_category_followers 
ON kols (primary_category) WHERE is_active = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kols_updated_scraped 
ON kols (updated_at, last_scraped) WHERE is_active = true;

-- Composite index for common filtering combinations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kols_composite_filters 
ON kols (platform, tier, primary_category, is_verified, is_brand_safe) 
WHERE is_active = true;

-- Performance index for KOL metrics (critical for POC2 matching)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kol_metrics_performance 
ON kol_metrics (kol_id, metrics_date DESC, follower_count, engagement_rate);

-- Index for recent metrics lookup
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kol_metrics_recent 
ON kol_metrics (kol_id, metrics_date DESC) 
WHERE metrics_date > CURRENT_DATE - INTERVAL '30 days';

-- Partial indexes for brand-safe KOLs (commonly filtered)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kols_brand_safe 
ON kols (tier, primary_category, follower_count) 
WHERE is_active = true AND is_brand_safe = true;

-- GIN index for JSONB operations (for dynamic filtering)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kols_secondary_categories_gin 
ON kols USING gin (secondary_categories);

-- Text search index for KOL discovery
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kols_search 
ON kols USING gin (to_tsvector('english', coalesce(display_name, '') || ' ' || coalesce(bio, '')));

-- Vector similarity index for content matching (POC2)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kols_content_embedding_ivfflat 
ON kols USING ivfflat (content_embedding vector_cosine_ops) WITH (lists = 100)
WHERE content_embedding IS NOT NULL;

-- Integration requests performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_integration_requests_tracking 
ON integration_requests (request_type, status, created_at DESC);

-- Content analysis indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kol_content_performance 
ON kol_content (kol_id, posted_at DESC, likes_count, comments_count);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kol_content_categories 
ON kol_content USING gin (content_categories);

-- Queue performance indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_status_priority 
ON jobs (status, priority DESC, created_at) WHERE status IN ('pending', 'running');

-- Foreign key performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kol_metrics_kol_id 
ON kol_metrics (kol_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_kol_content_kol_id 
ON kol_content (kol_id);

-- Statistics update for better query planning
ANALYZE kols;
ANALYZE kol_metrics;
ANALYZE kol_content;
ANALYZE integration_requests;

-- Materialized view for dashboard statistics (optional performance boost)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_kol_stats AS
SELECT 
    platform,
    tier,
    primary_category,
    COUNT(*) as total_kols,
    COUNT(*) FILTER (WHERE is_verified = true) as verified_kols,
    COUNT(*) FILTER (WHERE is_brand_safe = true) as brand_safe_kols,
    AVG(COALESCE((
        SELECT follower_count 
        FROM kol_metrics km 
        WHERE km.kol_id = k.id 
        ORDER BY metrics_date DESC 
        LIMIT 1
    ), 0)) as avg_followers,
    AVG(COALESCE((
        SELECT engagement_rate 
        FROM kol_metrics km 
        WHERE km.kol_id = k.id 
        ORDER BY metrics_date DESC 
        LIMIT 1
    ), 0)) as avg_engagement
FROM kols k
WHERE is_active = true
GROUP BY platform, tier, primary_category;

-- Create unique index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_kol_stats_unique 
ON mv_kol_stats (platform, tier, primary_category);

-- Refresh materialized view procedure
CREATE OR REPLACE FUNCTION refresh_kol_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_kol_stats;
END;
$$ LANGUAGE plpgsql;

-- Performance monitoring queries (for diagnostics)
/*
-- Query to find slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements 
WHERE query LIKE '%kols%' 
ORDER BY mean_time DESC 
LIMIT 10;

-- Index usage statistics
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes 
WHERE schemaname = 'public'
ORDER BY idx_tup_read DESC;

-- Table bloat check
SELECT 
    schemaname,
    tablename,
    n_tup_ins,
    n_tup_upd,
    n_tup_del,
    n_dead_tup
FROM pg_stat_user_tables 
WHERE schemaname = 'public'
ORDER BY n_dead_tup DESC;
*/