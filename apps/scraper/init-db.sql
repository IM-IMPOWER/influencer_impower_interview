-- AIDEV-NOTE: Database initialization script for KOL scraper service
-- Creates required extensions and basic database setup

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create basic indexes for performance (tables will be created by migrations)
-- This script runs before the application starts, so we only set up extensions

-- Set up database configuration for optimal performance
ALTER DATABASE kol_db SET shared_preload_libraries = 'pg_stat_statements';
ALTER DATABASE kol_db SET log_statement = 'all';
ALTER DATABASE kol_db SET log_min_duration_statement = 1000;

-- Create a basic health check function
CREATE OR REPLACE FUNCTION health_check() 
RETURNS TEXT AS $$
BEGIN
    RETURN 'OK';
END;
$$ LANGUAGE plpgsql;