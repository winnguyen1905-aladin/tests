#!/usr/bin/env python3
"""
SQLAlchemy Database Session Management for SAM3

Provides connection pooling, session factory, and helper functions
for SQLAlchemy with GeoAlchemy2 (PostGIS) and pgvector support.

This replaces the old Prisma-based approach with native SQLAlchemy
for better Flask/FastAPI integration.
"""

import os
import logging
import threading
from contextlib import contextmanager
from typing import Generator, Optional, Any, Dict

from sqlalchemy import create_engine, event, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


# Try to import vector type - works after pip install pgvector
try:
    from pgvector.sqlalchemy import Vector
    HAS_PGVECTOR = True
except ImportError:
    HAS_PGVECTOR = False
    Vector = None

# Try to import GeoAlchemy2
try:
    from geoalchemy2 import Geography, func
    HAS_GEOALCHEMY = True
except ImportError:
    HAS_GEOALCHEMY = False
    from sqlalchemy import func


class DatabaseConfig:
    """Configuration for database connection."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
    ):
        self.host = host or os.getenv("POSTGRES_HOST", "localhost")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = database or os.getenv("POSTGRES_DB", "sam3")
        self.user = user or os.getenv("POSTGRES_USER", "sam3user")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "sam3pass")
        self.pool_size = pool_size
        self.max_overflow = max_overflow

    @property
    def connection_string(self) -> str:
        """Get SQLAlchemy connection string."""
        return (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        )

    @property
    def safe_url(self) -> str:
        """Connection URL with password masked (safe for logging)."""
        return (
            f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.database}"
        )


class DatabaseManager:
    """
    Manages SQLAlchemy engine and sessions for SAM3.

    Provides:
    - Connection pooling
    - Automatic vector/geography type registration
    - Context manager for sessions
    - Health checks
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._connected

    def connect(self) -> bool:
        """Establish database connection and register types."""
        try:
            # Create engine with connection pooling
            self._engine = create_engine(
                self.config.connection_string,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,  # Verify connections before use
                echo=False,  # Set to True for SQL debugging
                connect_args={"connect_timeout": 5},  # Fail fast if unreachable
                hide_parameters=True,   # prevent credentials leaking into error messages
            )

            # Create required extensions once before any sessions are opened.
            if HAS_PGVECTOR or HAS_GEOALCHEMY:
                self._ensure_extensions()

            # Test connection
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.close()

            # Create session factory
            self._session_factory = sessionmaker(bind=self._engine)

            self._connected = True
            logger.info(f"Connected to PostgreSQL: {self.config.safe_url}")
            logger.info(f"Features enabled: pgvector={HAS_PGVECTOR}, GeoAlchemy2={HAS_GEOALCHEMY}")
            return True

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            logger.error("Install required: pip install sqlalchemy geoalchemy2 pgvector")
            self._connected = False
            return False

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            self._connected = False
            return False

    def _ensure_extensions(self) -> None:
        """Create pgvector and postgis extensions if not already present.
        Called once during connect(), before any session is created.
        """
        try:
            with self._engine.begin() as conn:   # auto-commits on exit
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            logger.info("Database extensions ensured: vector, postgis")
        except Exception as e:
            logger.warning(f"Could not create extensions (may require superuser): {e}")

    def _register_pgvector_types(self) -> None:
        """Register pgvector type with SQLAlchemy (deprecated – kept as no-op stub)."""
        pass

    def _register_postgis_functions(self) -> None:
        """Ensure PostGIS extension is available (deprecated – kept as no-op stub)."""
        pass

    def disconnect(self) -> None:
        """Close database connection."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._connected = False
            logger.info("Disconnected from PostgreSQL")

    def get_session(self) -> Session:
        """Get a new database session."""
        if self._session_factory is None:
            if not self.connect():
                raise RuntimeError("Failed to connect to database")
        return self._session_factory()

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()

    @contextmanager
    def transactional_session(self, session: Session) -> Generator[Session, None, None]:
        """Nested transaction context manager."""
        with session.begin_nested():
            yield session

    def execute_raw(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute raw SQL query."""
        with self.session() as session:
            result = session.execute(text(query), params or {})
            return result

    def execute_raw_with_connection(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute raw SQL with direct connection (for vector operations)."""
        if self._engine is None:
            raise RuntimeError("Not connected to database")

        with self._engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            if query.strip().upper().startswith("SELECT"):
                return result.fetchall()
            conn.commit()
            return result

    def get_table_names(self) -> list:
        """Get list of tables in database."""
        with self._engine.connect() as conn:
            result = conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            ))
            return [row[0] for row in result]

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        # Use parameterized query to prevent SQL injection
        with self._engine.connect() as conn:
            result = conn.execute(text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = :table_name"
            ), {"table_name": table_name})
            return result.fetchone() is not None

    def health_check(self) -> bool:
        """Check database health."""
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.close()

                # Check extensions
                result = conn.execute(text(
                    "SELECT extname FROM pg_extension WHERE extname IN ('vector', 'postgis')"
                ))
                extensions = {row[0] for row in result}
                result.close()

                logger.info(f"Health check: OK (extensions: {extensions})")
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Global database manager instance with thread-safe initialization
_db_manager: Optional[DatabaseManager] = None
_db_manager_lock = threading.Lock()


def get_db_manager(config: Optional[DatabaseConfig] = None) -> Optional[DatabaseManager]:
    """Get or create the global database manager.

    Thread-safe implementation using double-checked locking.
    Returns None if the database has not been initialized (e.g. after shutdown_db).
    """
    global _db_manager
    if _db_manager is None:
        with _db_manager_lock:
            if _db_manager is None and config is not None:
                _db_manager = DatabaseManager(config)
    return _db_manager


def init_db(
    host: Optional[str] = None,
    port: Optional[int] = None,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    pool_size: int = 10,
    max_overflow: int = 20,
    app_config: Optional[Any] = None,
) -> DatabaseManager:
    """Initialize database connection.

    Args:
        host, port, database, user, password: Override connection params.
        pool_size, max_overflow: Pool settings.
        app_config: Optional object with postgres_host, postgres_port, etc.
                    When provided, used for connection params when others are None.
    """
    if app_config is not None:
        host = host or getattr(app_config, "postgres_host", None)
        port = port if port is not None else getattr(app_config, "postgres_port", None)
        database = database or getattr(app_config, "postgres_db", None)
        user = user or getattr(app_config, "postgres_user", None)
        password = password or getattr(app_config, "postgres_password", None)
        pool_size = getattr(app_config, "postgres_pool_size", pool_size)
        max_overflow = getattr(app_config, "postgres_max_overflow", max_overflow)
    config = DatabaseConfig(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        pool_size=pool_size,
        max_overflow=max_overflow,
    )
    with _db_manager_lock:
        global _db_manager
        _db_manager = DatabaseManager(config)
    manager = _db_manager
    manager.connect()
    return manager


def shutdown_db() -> None:
    """Disconnect and clear the global database manager (for app shutdown).

    After this, get_session() will fail until init_db() is called again.
    """
    global _db_manager
    with _db_manager_lock:
        if _db_manager is not None:
            _db_manager.disconnect()
            _db_manager = None
            logger.info("Database manager shut down")


def get_session() -> Session:
    """Get a database session (convenience function).

    Raises:
        RuntimeError: If the database has not been initialized (call init_db first).
    """
    manager = get_db_manager()
    if manager is None:
        raise RuntimeError("Database not initialized; call init_db() first")
    return manager.get_session()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session via context manager (convenience function)."""
    manager = get_db_manager()
    if manager is None:
        raise RuntimeError("Database not initialized; call init_db() first")
    with manager.session() as session:
        yield session


# Import models after base is defined
def get_models():
    """Get all ORM models."""
    from src.repository.entityModels import Base, Tree, TreeEvidence
    return Base, Tree, TreeEvidence


def create_tables(if_not_exists: bool = True) -> None:
    """Create all tables in database."""
    from src.repository.spatialEntityModels import FarmZone  # noqa: F401
    from src.repository.entityModels import Base
    manager = get_db_manager()
    if manager is None or manager._engine is None:
        raise RuntimeError(
            "Database not initialized; call init_db() before create_tables()."
        )
    Base.metadata.create_all(manager._engine, checkfirst=if_not_exists)
    logger.info("Tables created successfully")


def drop_tables() -> None:
    """Drop all tables in database."""
    from src.repository.spatialEntityModels import FarmZone  # noqa: F401
    from src.repository.entityModels import Base
    manager = get_db_manager()
    if manager is None or manager._engine is None:
        raise RuntimeError(
            "Database not initialized; call init_db() before drop_tables()."
        )
    Base.metadata.drop_all(manager._engine)
    logger.info("Tables dropped successfully")


def migrate_vector_dimension(
    target_dim: int,
    *,
    recreate_hnsw: bool = True,
) -> bool:
    """Migrate tree_evidences.global_vector column to a new halfvec dimension.

    Use this when ``POSTGRES_VECTOR_DIM`` / your DINO model output no longer matches
    the column (e.g. ``halfvec(384)`` in DB but the model emits 768).

    Drops any indexes that reference ``global_vector`` (required before ``ALTER``),
    clears stored vectors (pgvector cannot convert between widths in-place),
    then sets the column to ``halfvec(target_dim)``. Optionally recreates an HNSW
    cosine index for similarity search.

    Args:
        target_dim: Target dimension (e.g. 384 for dinov3-vitb16, 768 for
            dinov2-vitb14 / dinov3-convnext-small).
        recreate_hnsw: If True and ``postgres_vector_index_type`` is ``hnsw``,
            create ``idx_tree_evidences_global_vector_hnsw`` after migration.

    Returns:
        True if migration was successful or already at ``target_dim``, False on error.
    """
    import re

    manager = get_db_manager()
    if manager is None or manager._engine is None:
        logger.error("Database not initialized; call init_db() before migrate_vector_dimension().")
        return False

    if target_dim < 1 or target_dim > 65535:
        logger.error(f"Invalid target_dim: {target_dim} (must be 1..65535)")
        return False

    with manager.session() as session:
        try:
            result = session.execute(text("""
                SELECT pg_catalog.format_type(a.atttypid, a.atttypmod) AS col_type
                FROM pg_catalog.pg_attribute a
                WHERE a.attrelid = 'tree_evidences'::regclass
                  AND a.attname = 'global_vector'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
            """))
            row = result.fetchone()
            if not row:
                logger.warning("global_vector column not found in tree_evidences table")
                return False

            current_type = (row[0] or "").strip()
            logger.info(f"Current global_vector type: {current_type}")

            match = re.match(r"^(halfvec|vector)\((\d+)\)$", current_type, re.IGNORECASE)
            if not match:
                logger.warning(
                    f"global_vector is not a typed halfvec/vector with dimension: {current_type}"
                )
                return False

            current_dim = int(match.group(2))
            if current_dim == target_dim:
                logger.info(f"global_vector already has target dimension {target_dim}, no migration needed")
                return True

            logger.info(f"Migrating global_vector from {current_type} to halfvec({target_dim})")

            idx_rows = session.execute(
                text("""
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND tablename = 'tree_evidences'
                      AND indexdef ILIKE '%global_vector%'
                """)
            ).fetchall()
            for (index_name,) in idx_rows:
                if not index_name:
                    continue
                # Identifier from pg_indexes; quote for safety
                session.execute(text(f'DROP INDEX IF EXISTS "{index_name}"'))
                logger.info(f"Dropped index {index_name} (referenced global_vector)")

            session.execute(
                text("UPDATE tree_evidences SET global_vector = NULL WHERE global_vector IS NOT NULL")
            )
            logger.info("Cleared existing vector data for migration")

            session.execute(
                text(
                    f"ALTER TABLE tree_evidences "
                    f"ALTER COLUMN global_vector TYPE halfvec({target_dim})"
                )
            )
            logger.info(f"Successfully altered global_vector to halfvec({target_dim})")

            try:
                session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception:
                pass

            if recreate_hnsw and HAS_PGVECTOR:
                try:
                    from src.config.appConfig import get_config

                    cfg = get_config()
                    if getattr(cfg, "postgres_vector_index_type", "").lower() == "hnsw":
                        m = int(cfg.postgres_vector_m)
                        ef = int(cfg.postgres_vector_ef_construction)
                        session.execute(
                            text(
                                f"""
                                CREATE INDEX IF NOT EXISTS idx_tree_evidences_global_vector_hnsw
                                ON tree_evidences
                                USING hnsw (global_vector halfvec_cosine_ops)
                                WITH (m = {m}, ef_construction = {ef})
                                """
                            )
                        )
                        logger.info(
                            "Created HNSW index idx_tree_evidences_global_vector_hnsw "
                            f"(m={m}, ef_construction={ef})"
                        )
                except Exception as idx_err:
                    logger.warning(
                        f"HNSW index was not recreated (similarity search may be slower): {idx_err}"
                    )

            return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            session.rollback()
            return False


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Database Manager Demo")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", default="sam3")
    parser.add_argument(
        "--migrate-vector-dim",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Alter tree_evidences.global_vector to halfvec(N). "
            "Set POSTGRES_VECTOR_DIM=N to match. Clears existing vectors."
        ),
    )
    parser.add_argument(
        "--no-recreate-hnsw",
        action="store_true",
        help="With --migrate-vector-dim, skip creating idx_tree_evidences_global_vector_hnsw",
    )
    args = parser.parse_args()

    # Initialize
    manager = init_db(
        host=args.host,
        port=args.port,
        database=args.database,
    )

    if manager.is_connected:
        print(f"Connected to PostgreSQL: {args.host}:{args.port}/{args.database}")
        print(f"Features: pgvector={HAS_PGVECTOR}, GeoAlchemy2={HAS_GEOALCHEMY}")

        if args.migrate_vector_dim is not None:
            ok = migrate_vector_dimension(
                args.migrate_vector_dim,
                recreate_hnsw=not args.no_recreate_hnsw,
            )
            print(f"migrate_vector_dimension({args.migrate_vector_dim}): {'ok' if ok else 'failed'}")
            if not ok:
                manager.disconnect()
                raise SystemExit(1)

        # List tables
        tables = manager.get_table_names()
        print(f"Tables: {tables}")

        # Health check
        print(f"Health: {manager.health_check()}")

        manager.disconnect()
    else:
        print("Failed to connect to PostgreSQL")
