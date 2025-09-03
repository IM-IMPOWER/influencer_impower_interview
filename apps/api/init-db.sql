-- AIDEV-NOTE: Database initialization script for PostgreSQL with extensions

-- Create extensions required for KOL platform
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create indexes for full-text search
CREATE EXTENSION IF NOT EXISTS "unaccent";

-- Set up vector similarity search configuration
-- AIDEV-NOTE: This configures pgvector for optimal performance with sentence transformers
ALTER SYSTEM SET shared_preload_libraries = 'vector';

-- Create custom functions for vector similarity
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector) RETURNS float AS $$
BEGIN
    RETURN (a <=> b);
END;
$$ LANGUAGE plpgsql IMMUTABLE;