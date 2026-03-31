-- scripts/init-pgcrypto.sql
-- Runs once on first container boot (via docker-entrypoint-initdb.d)
-- postgis + postgis_topology are already enabled by the postgis/postgis image.
-- We just need pgcrypto for gen_random_uuid().
CREATE EXTENSION IF NOT EXISTS pgcrypto;
