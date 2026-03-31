#!/usr/bin/env python3
"""
Migration script: make trees.row_idx / trees.col_idx nullable.

This migration also removes obsolete column defaults and restores the
``idx_trees_grid`` composite index on ``(farm_id, row_idx, col_idx)``.

Usage:
    python scripts/migrate_tree_grid_nullable.py [--dry-run]
"""

import argparse
import logging
import os
import sys

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


def get_db_config() -> dict:
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "sam3"),
        "user": os.getenv("POSTGRES_USER", "sam3user"),
        "password": os.getenv("POSTGRES_PASSWORD", "sam3pass"),
    }


def planned_sql() -> list[str]:
    return [
        "ALTER TABLE trees ALTER COLUMN row_idx DROP DEFAULT",
        "ALTER TABLE trees ALTER COLUMN col_idx DROP DEFAULT",
        "ALTER TABLE trees ALTER COLUMN row_idx DROP NOT NULL",
        "ALTER TABLE trees ALTER COLUMN col_idx DROP NOT NULL",
        "CREATE INDEX CONCURRENTLY idx_trees_grid ON trees (farm_id, row_idx, col_idx)",
    ]


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


def column_info(cursor, table: str, column: str) -> tuple[str, str | None]:
    cursor.execute(
        """
        SELECT is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
        """,
        (table, column),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"Column '{table}.{column}' does not exist")
    is_nullable, column_default = row
    return str(is_nullable), column_default


def index_exists(cursor, table: str, index_name: str) -> bool:
    cursor.execute(
        """
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = %s
          AND indexname = %s
        """,
        (table, index_name),
    )
    return cursor.fetchone() is not None


def migrate(dry_run: bool = False) -> bool:
    if dry_run:
        logger.info("DRY-RUN mode - planned SQL:")
        for statement in planned_sql():
            logger.info("  %s", statement)
        logger.info("Dry-run prints the desired migration SQL; live mode performs idempotency checks.")
        return True

    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        return False

    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(**get_db_config())
        conn.autocommit = True
        cursor = conn.cursor()

        table = "trees"
        if not table_exists(cursor, table):
            logger.error("Table '%s' does not exist", table)
            return False

        statements: list[str] = []
        for column in ("row_idx", "col_idx"):
            is_nullable, default = column_info(cursor, table, column)
            if default is not None:
                statements.append(f"ALTER TABLE {table} ALTER COLUMN {column} DROP DEFAULT")
            if is_nullable != "YES":
                statements.append(f"ALTER TABLE {table} ALTER COLUMN {column} DROP NOT NULL")

        if not index_exists(cursor, table, "idx_trees_grid"):
            statements.append(
                "CREATE INDEX CONCURRENTLY idx_trees_grid ON trees (farm_id, row_idx, col_idx)"
            )

        if not statements:
            logger.info("No migration needed; trees grid columns and index already match the target schema")
        else:
            for statement in statements:
                logger.info("Executing: %s", statement)
                cursor.execute(statement)

        row_nullable, row_default = column_info(cursor, table, "row_idx")
        col_nullable, col_default = column_info(cursor, table, "col_idx")
        has_index = index_exists(cursor, table, "idx_trees_grid")

        logger.info(
            "Verification - row_idx nullable=%s default=%s",
            row_nullable,
            row_default,
        )
        logger.info(
            "Verification - col_idx nullable=%s default=%s",
            col_nullable,
            col_default,
        )
        logger.info("Verification - idx_trees_grid exists=%s", has_index)

        return row_nullable == "YES" and col_nullable == "YES" and has_index

    except psycopg2.OperationalError as exc:
        logger.error("Database connection failed: %s", exc)
        logger.error("Check PostgreSQL credentials and ensure the server is running")
        return False
    except Exception as exc:
        logger.error("Migration failed: %s", exc)
        return False
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make trees.row_idx / trees.col_idx nullable and restore idx_trees_grid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the intended SQL without connecting to PostgreSQL.",
    )
    args = parser.parse_args()

    success = migrate(dry_run=args.dry_run)
    if success:
        logger.info("✓ Migration completed successfully")
        sys.exit(0)

    logger.error("✗ Migration failed")
    sys.exit(1)


if __name__ == "__main__":
    main()
