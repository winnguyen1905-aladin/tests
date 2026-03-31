#!/usr/bin/env python3
"""
Migration script: add row_idx / col_idx columns to farm_zones.

These columns are nullable — they can be populated later when a chessboard
segmentation is applied to the farm polygon.

Usage:
    python scripts/migrate_farm_zone_grid_idx.py [--dry-run]

    # Or via Makefile:
    make migrate-farmzone-grid
"""

import os
import sys
import argparse
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def get_db_config() -> dict:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "sam3"),
        "user": os.getenv("POSTGRES_USER", "sam3user"),
        "password": os.getenv("POSTGRES_PASSWORD", "sam3pass"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def table_exists(cursor, table: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table,),
    )
    return cursor.fetchone() is not None


def column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s AND column_name = %s
        """,
        (table, column),
    )
    return cursor.fetchone() is not None


def migrate(dry_run: bool = False) -> bool:
    """Add row_idx / col_idx (nullable SMALLINT) to farm_zones.

    If the farm_zones table does not yet exist at all, the table is created
    first (including all existing FarmZone columns) so the migration is safe
    on a fresh schema too.
    """
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        return False

    config = get_db_config()
    conn = psycopg2.connect(**config)
    conn.autocommit = True
    cursor = conn.cursor()

    table = "farm_zones"

    # ── 1. Create table if it doesn't exist ──────────────────────────────────
    if not table_exists(cursor, table):
        if dry_run:
            logger.info("[DRY-RUN] Would CREATE TABLE farm_zones …")
        else:
            cursor.execute("""
                CREATE TABLE farm_zones (
                    farm_id      VARCHAR(50)  NOT NULL PRIMARY KEY,
                    owner_did    VARCHAR(100) NOT NULL,
                    region_code  VARCHAR(10)  NOT NULL,
                    farm_name    VARCHAR(255),
                    boundary     TEXT         NOT NULL,
                    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
                    row_idx      SMALLINT,
                    col_idx      SMALLINT
                )
            """)
            cursor.execute("""
                CREATE INDEX idx_farm_zones_boundary ON farm_zones
                    USING GIST (ST_GeomFromEWKB(boundary::geometry))
            """)
            cursor.execute("""
                CREATE INDEX idx_farm_zones_region_code ON farm_zones (region_code)
            """)
            cursor.execute("""
                CREATE INDEX idx_farm_zones_owner_did ON farm_zones (owner_did)
            """)
            cursor.execute("""
                CREATE INDEX idx_farm_zones_updated_at ON farm_zones (updated_at)
            """)
            logger.info(f"[{table}] Table created with all columns including row_idx/col_idx")
    else:
        # ── 2. Add columns if table already exists ─────────────────────────────
        columns = [
            ("row_idx", "SMALLINT"),
            ("col_idx", "SMALLINT"),
        ]
        for col_name, col_type in columns:
            if column_exists(cursor, table, col_name):
                logger.info(f"[{table}] column '{col_name}' already exists — skipping")
            else:
                sql = f'ALTER TABLE {table} ADD COLUMN {col_name} {col_type}'
                if dry_run:
                    logger.info(f"[DRY-RUN] Would execute: {sql}")
                else:
                    cursor.execute(sql)
                    logger.info(f"[{table}] Added column '{col_name}' ({col_type}) NULL")

    # Verify
    cursor.execute(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = %s
          AND column_name IN ('row_idx', 'col_idx')
        ORDER BY column_name
        """,
        (table,),
    )
    rows = cursor.fetchall()
    logger.info("Verification — farm_zones row_idx/col_idx:")
    for row_name, row_type, row_nullable in rows:
        logger.info(f"  {row_name}: {row_type} nullable={row_nullable}")

    cursor.close()
    conn.close()
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Migrate farm_zones: add nullable row_idx/col_idx columns.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SQL without executing it.",
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY-RUN mode — no changes will be applied.")

    success = migrate(dry_run=args.dry_run)
    if success:
        action = "DRY-RUN" if args.dry_run else "Migration"
        logger.info(f"✓ {action} completed successfully")
        sys.exit(0)
    else:
        logger.error("✗ Migration failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
