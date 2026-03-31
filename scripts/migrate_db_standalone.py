#!/usr/bin/env python3
"""
Standalone migration script for tree_evidences.global_vector column.

Run this AFTER activating your Python environment with psycopg2:
    conda activate your_env  # or source venv/bin/activate
    python scripts/migrate_db_standalone.py

Or install dependencies first:
    pip install psycopg2-binary
"""

import os
import sys
import argparse
import logging

# Load .env file
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


def get_db_config():
    """Get database config from environment."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "sam3"),
        "user": os.getenv("POSTGRES_USER", "sam3user"),
        "password": os.getenv("POSTGRES_PASSWORD", "sam3pass"),
    }


def migrate_vector_dimension(target_dim: int) -> bool:
    """Migrate tree_evidences.global_vector column to new dimension."""
    import re
    
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
        return False
    
    config = get_db_config()
    
    try:
        conn = psycopg2.connect(**config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Step 1: Check current column type
        cursor.execute("""
            SELECT atttypname
            FROM pg_attribute
            JOIN pg_type ON atttypid = pg_type.oid
            WHERE attrelid = 'tree_evidences'::regclass
            AND attname = 'global_vector'
        """)
        row = cursor.fetchone()
        
        if not row:
            logger.error("global_vector column not found in tree_evidences table")
            return False
        
        current_type = row[0]
        logger.info(f"Current global_vector type: {current_type}")
        
        # Check if it's a halfvec type
        match = re.match(r'halfvec\((\d+)\)', current_type)
        if not match:
            logger.error(f"global_vector is not a halfvec type: {current_type}")
            return False
        
        current_dim = int(match.group(1))
        
        if current_dim == target_dim:
            logger.info(f"global_vector already has target dimension {target_dim}, no migration needed")
            cursor.close()
            conn.close()
            return True
        
        # Step 2: Clear existing vectors (pgvector requires empty column for type change)
        logger.info(f"Migrating from halfvec({current_dim}) to halfvec({target_dim})")
        logger.info("Clearing existing vector data...")
        cursor.execute("UPDATE tree_evidences SET global_vector = NULL WHERE global_vector IS NOT NULL")
        logger.info(f"Cleared vectors from tree_evidences table")
        
        # Step 3: Alter the column type
        logger.info(f"Altering column type to halfvec({target_dim})...")
        cursor.execute(f"ALTER TABLE tree_evidences ALTER COLUMN global_vector TYPE halfvec({target_dim})")
        logger.info(f"Successfully altered global_vector to halfvec({target_dim})")
        
        # Step 4: Verify
        cursor.execute("""
            SELECT atttypname
            FROM pg_attribute
            JOIN pg_type ON atttypid = pg_type.oid
            WHERE attrelid = 'tree_evidences'::regclass
            AND attname = 'global_vector'
        """)
        new_type = cursor.fetchone()[0]
        logger.info(f"New global_vector type: {new_type}")
        
        cursor.close()
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("Check your database credentials and ensure PostgreSQL is running")
        return False
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Migrate tree_evidences.global_vector to new dimension")
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Target vector dimension (default: POSTGRES_VECTOR_DIM env var or 768)",
    )
    args = parser.parse_args()
    
    target_dim = args.dim
    if target_dim is None:
        target_dim = int(os.getenv("POSTGRES_VECTOR_DIM", "768"))
    
    logger.info(f"Starting migration to dimension: {target_dim}")
    
    success = migrate_vector_dimension(target_dim)
    
    if success:
        logger.info(f"✓ Migration to halfvec({target_dim}) completed successfully")
        sys.exit(0)
    else:
        logger.error(f"✗ Migration failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
