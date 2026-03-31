-- scripts/init-pgvector-postgis.sql
-- PostgreSQL schema for SAM3 with pgvector and PostGIS
-- Runs on first container boot via docker-entrypoint-initdb.d
--
-- This script creates:
--   1. pgvector extension for vector similarity search
--   2. trees table with PostGIS geography column
--   3. tree_evidences table with pgvector halfvec column
--   4. Indexes for efficient queries

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- TREES TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS trees (
    id VARCHAR(255) PRIMARY KEY,
    region_code VARCHAR(50) NOT NULL DEFAULT 'default',
    farm_id VARCHAR(255) NOT NULL DEFAULT 'default_farm',
    geohash_7 VARCHAR(20) NOT NULL DEFAULT 'default',
    location geography(POINT, 4326),
    row_idx INTEGER NOT NULL DEFAULT 0,
    col_idx INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- TREE EVIDENCES TABLE
-- ============================================================

-- Note: VECTOR_DIM is set to 1280 for dinov3-vith16plus
-- If using a different DINO model, adjust the halfvec dimension

CREATE TABLE IF NOT EXISTS tree_evidences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tree_id VARCHAR(255) NOT NULL REFERENCES trees(id) ON DELETE CASCADE,
    region_code VARCHAR(50) NOT NULL DEFAULT 'default',
    global_vector halfvec(1280),
    storage_cid TEXT,
    evidence_hash VARCHAR(255),
    is_c2pa_verified BOOLEAN DEFAULT FALSE,
    camera_heading INTEGER,
    camera_pitch INTEGER,
    camera_roll INTEGER,
    raw_telemetry JSONB,
    metadata JSONB,
    location geography(POINT, 4326),
    captured_at INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================
-- INDEXES
-- ============================================================

-- Index on tree_id for fast lookups
CREATE INDEX IF NOT EXISTS idx_tree_evidences_tree_id
ON tree_evidences(tree_id);

-- Index on region_code
CREATE INDEX IF NOT EXISTS idx_tree_evidences_region_code
ON tree_evidences(region_code);

-- Index on captured_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_tree_evidences_captured_at
ON tree_evidences(captured_at DESC);

-- HNSW index for vector similarity search (recommended for production)
-- Parameters:
--   m: number of connections per layer (default: 16)
--   ef_construction: search width during index build (default: 64)
-- Higher values = better recall, slower build, more memory
CREATE INDEX IF NOT EXISTS idx_tree_evidences_vector_hnsw
ON tree_evidences
USING hnsw (global_vector halfvec_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- GiST index for PostGIS spatial queries
CREATE INDEX IF NOT EXISTS idx_tree_evidences_location_gist
ON tree_evidences USING GIST (location);

-- Index on trees location
CREATE INDEX IF NOT EXISTS idx_trees_location_gist
ON trees USING GIST (location);

-- ============================================================
-- HELPER FUNCTIONS
-- ============================================================

-- Function to convert longitude/latitude to geography point
CREATE OR REPLACE FUNCTION make_point_geog(lon float, lat float)
RETURNS geography(Point, 4326)
LANGUAGE sql
AS $$
  SELECT ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography;
$$;

-- Function to calculate distance in meters between two points
CREATE OR REPLACE FUNCTION distance_meters(p1 geography, p2 geography)
RETURNS float
LANGUAGE sql
AS $$
  SELECT ST_Distance(p1, p2);
$$;

-- ============================================================
-- COMMENTS
-- ============================================================

COMMENT ON EXTENSION vector IS 'Vector similarity search extension for pgvector';
COMMENT ON COLUMN trees.location IS 'PostGIS geography Point for spatial queries';
COMMENT ON COLUMN tree_evidences.global_vector IS 'pgvector halfvec(1280) for DINOv3 embeddings';
