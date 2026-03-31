#!/usr/bin/env python3
"""
SAM3 Repository Module

Provides data access layer for the SAM3 Tree Identification System.

Modules:
- database: SQLAlchemy connection and session management
- models: SQLAlchemy ORM models (Tree, TreeEvidence)
- sqlalchemyRepository: SQLAlchemy ORM-based repository with pgvector and PostGIS
- milvusRepository: Milvus vector database operations
- minioRepository: MinIO/S3 object storage operations
"""

# Database connection and session management
from src.repository.databaseManager import (
    DatabaseConfig,
    DatabaseManager,
    get_db_manager,
    init_db,
    shutdown_db,
    get_session,
    get_db_session,
    create_tables,
    drop_tables,
    migrate_vector_dimension,
)

# SQLAlchemy ORM models
from src.repository.entityModels import (
    Base,
    Tree,
    TreeEvidence,
    get_column_types,
)

# SQLAlchemy ORM repository
from src.repository.sqlalchemyRepository import (
    SQLAlchemyORMRepository,
    TreeRecord,
    TreeEvidenceRecord,
    VectorSearchResult,
    GeoSearchResult,
    create_sqlalchemy_repository,
    get_sqlalchemy_repository,
)

__all__ = [
    # Database management
    "DatabaseConfig",
    "DatabaseManager",
    "get_db_manager",
    "init_db",
    "shutdown_db",
    "get_session",
    "get_db_session",
    "create_tables",
    "drop_tables",
    "migrate_vector_dimension",
    # Models
    "Base",
    "Tree",
    "TreeEvidence",
    "get_column_types",
    # ORM repository
    "SQLAlchemyORMRepository",
    "TreeRecord",
    "TreeEvidenceRecord",
    "VectorSearchResult",
    "GeoSearchResult",
    "create_sqlalchemy_repository",
    "get_sqlalchemy_repository",
]
