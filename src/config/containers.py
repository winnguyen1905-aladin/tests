#!/usr/bin/env python3
"""
Dependency Injection Container for SAM3

Defines all providers for repositories, processors, and services following
dependency_injector patterns.

Usage:
    # Wire modules for @inject pattern:
    from containers import container
    container.wire(modules=[
        "src.service.preprocessorService",
        "src.service.ingestionService",
        "src.service.verificationService",
        "src.service.hierarchicalMatchingService",
    ])
    
    # Get a service instance (services use @inject):
    service = container.ingestion_service()
    
    # In FastAPI with wiring:
    from src.api.dependencies import get_ingestion_service
    
    @app.get("/health")
    def health(service: IngestionService = Depends(get_ingestion_service)):
        return service.health_check()
"""

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from typing import Optional, Any, Dict, Callable

# =============================================================================
# Configuration
# =============================================================================
from src.config.appConfig import AppConfig, get_config


def _get_app_config() -> AppConfig:
    """Get application configuration singleton."""
    return get_config()


# =============================================================================
# Repositories
# =============================================================================
from src.repository.sqlalchemyRepository import SQLAlchemyORMRepository
from src.repository.milvusRepository import MilvusRepository, MilvusConfig
from src.repository.minioRepository import MinIORepository, MinIOConfig


def _create_milvus_config() -> MilvusConfig:
    """Create Milvus config from AppConfig."""
    cfg = get_config()
    return MilvusConfig(
        uri=cfg.milvus_uri,
        collection_name=cfg.milvus_collection,
        vector_dim=cfg.milvus_vector_dim,
        nprobe=cfg.milvus_search_nprobe,
        verbose=False,
    )


def _create_minio_config() -> MinIOConfig:
    """Create MinIO config from AppConfig."""
    cfg = get_config()
    return MinIOConfig(
        endpoint=cfg.minio_endpoint,
        access_key=cfg.minio_access_key,
        secret_key=cfg.minio_secret_key,
        bucket=cfg.minio_bucket,
        secure=cfg.minio_secure,
        verbose=False,
    )


# =============================================================================
# Processors
# =============================================================================
from src.processor.dinoProcessor import DinoProcessor, DinoConfig
from src.processor.superPointProcessor import SuperPointProcessor, SuperPointConfig
from src.processor.lightGlueProcessor import LightGlueProcessor, LightGlueConfig
from src.processor.hierarchicalMatcher import HierarchicalMatcher, HierarchicalMatcherConfig


def _create_dino_config() -> DinoConfig:
    """Create Dino config from AppConfig."""
    cfg = get_config()
    return DinoConfig(
        model_type=cfg.dino_model_type,
        device=cfg.dino_device,
        image_size=cfg.dino_image_size,
        hf_token=cfg.dino_hf_token,
        use_multi_gpu=cfg.dino_use_multi_gpu,
        gpu_ids=cfg.dino_gpu_ids,
        enable_memory_optimization=cfg.dino_enable_memory_optimization,
        use_gradient_checkpointing=cfg.dino_use_gradient_checkpointing,
        use_4bit_quantization=cfg.dino_use_4bit_quantization,
        verbose=cfg.verbose,
    )


def _create_superpoint_config() -> SuperPointConfig:
    """Create SuperPoint config from AppConfig."""
    cfg = get_config()
    return SuperPointConfig(
        max_keypoints=cfg.sp_max_keypoints,
        max_dimension=cfg.sp_max_dimension,
        device=cfg.sp_device,
        verbose=cfg.verbose,
    )


def _create_lightglue_config() -> LightGlueConfig:
    """Create LightGlue config from AppConfig."""
    cfg = get_config()
    return LightGlueConfig(
        device=cfg.lg_device,
        filter_threshold=cfg.lg_filter_threshold,
        verbose=cfg.verbose,
    )


def _create_hierarchical_matcher_config() -> HierarchicalMatcherConfig:
    """Create HierarchicalMatcher config from AppConfig."""
    cfg = get_config()
    return HierarchicalMatcherConfig.from_app_config(cfg)


# =============================================================================
# Services (using @inject decorator pattern)
# =============================================================================
from src.service.preprocessorService import PreprocessorService
from src.service.ingestionService import IngestionService
from src.service.verificationService import VerificationService
from src.service.hierarchicalMatchingService import HierarchicalMatchingService
from src.utils.matchingStrategy import MatchingStrategy, MatchingStrategyConfig


# =============================================================================
# Container Definition
# =============================================================================
class Container(containers.DeclarativeContainer):
    """Dependency injection container for SAM3 application.

    Provider Types:
    - Configuration: Application settings
    - Singleton: Expensive to initialize, shared across app (processors, config)
    - Factory: Stateless, created on each request (repositories, services)

    Wiring:
    To use with FastAPI and @inject decorators, add this to main.py:
        from containers import container
        
        container.wire(modules=[
            "src.service.preprocessorService",
            "src.service.ingestionService",
            "src.service.verificationService",
            "src.service.hierarchicalMatchingService",
        ])
    
    Or use the convenience function:
        from containers import wire_services
        wire_services()
    """

    # ========================================================================
    # Configuration - Singleton
    # ========================================================================
    app_config = providers.Singleton(_get_app_config)

    # ========================================================================
    # Repository Configs - Factory
    # ========================================================================
    milvus_config = providers.Factory(_create_milvus_config)
    minio_config = providers.Factory(_create_minio_config)

    # ========================================================================
    # Repositories - Factory (stateless, new instance per request)
    # ========================================================================
    milvus_repo = providers.Factory(
        MilvusRepository,
        config=milvus_config,
    )

    minio_repo = providers.Factory(
        MinIORepository,
        config=minio_config,
    )

    sqlalchemy_repo = providers.Factory(
        SQLAlchemyORMRepository,
    )

    # ========================================================================
    # Processor Configs - Factory
    # ========================================================================
    dino_config = providers.Factory(_create_dino_config)
    superpoint_config = providers.Factory(_create_superpoint_config)
    lightglue_config = providers.Factory(_create_lightglue_config)
    hierarchical_matcher_config = providers.Factory(_create_hierarchical_matcher_config)

    # ========================================================================
    # Processors - Singleton (stateful, expensive to init, shared)
    # All receive config so they initialize with correct settings.
    # ========================================================================
    dino_processor = providers.Singleton(
        DinoProcessor,
        config=dino_config,
    )

    superpoint_processor = providers.Singleton(
        SuperPointProcessor,
        config=superpoint_config,
    )

    lightglue_processor = providers.Singleton(
        LightGlueProcessor,
        config=lightglue_config,
    )

    # ========================================================================
    # Preprocessor - Singleton (uses @inject with Provide[])
    # ========================================================================
    preprocessor_service = providers.Singleton(
        PreprocessorService,
        app_config=app_config,
        background_color="black",
    )

    # ========================================================================
    # Services - Factory (stateless business logic, use @inject)
    # ========================================================================
    ingestion_service = providers.Factory(
        IngestionService,
        preprocessor=preprocessor_service,
        app_config=app_config,
        dino_processor=dino_processor,
        superpoint_processor=superpoint_processor,
        postgres_repo=sqlalchemy_repo,
        minio_repo=minio_repo,
    )

    verification_service = providers.Factory(
        VerificationService,
        preprocessor=preprocessor_service,
        app_config=app_config,
        dino_processor=dino_processor,
        superpoint_processor=superpoint_processor,
        lightglue_processor=lightglue_processor,
        milvus_repo=milvus_repo,
        postgres_repo=sqlalchemy_repo,
        minio_repo=minio_repo,
        verbose=app_config,
    )

    # Hierarchical Matcher - Singleton (expensive to init, shared across requests)
    # Uses config from AppConfig via HierarchicalMatcherConfig
    hierarchical_matcher = providers.Singleton(
        HierarchicalMatcher,
        config=hierarchical_matcher_config,
    )

    hierarchical_matching_service = providers.Factory(
        HierarchicalMatchingService,
        postgres_repo=sqlalchemy_repo,
        minio_repo=minio_repo,
        matcher=hierarchical_matcher,
    )

    identification_service = providers.Factory(
        preprocessor=preprocessor_service,
        app_config=app_config,
        dino_processor=dino_processor,
        superpoint_processor=superpoint_processor,
        lightglue_processor=lightglue_processor,
        milvus_repo=milvus_repo,
        minio_repo=minio_repo,
    )


# ========================================================================
# Singleton container instance
# ========================================================================
container = Container()


def initialize_application_resources() -> None:
    """Initialize DB and other resources at FastAPI startup.

    **Do not** define ``init_resources`` on ``Container``: the compiled
    ``DynamicContainer`` already exposes ``init_resources`` for
    ``providers.Resource`` wiring, so a custom method would never run and
    ``init_db()`` would be skipped (PostgreSQL then fails with
    "Database not initialized").
    """
    from src.repository.databaseManager import init_db

    cfg = container.app_config()
    init_db(app_config=cfg)


def shutdown_application_resources() -> None:
    """Release DB and GPU resources at app shutdown."""
    from src.repository.databaseManager import shutdown_db
    from src.api.helpers import cleanup_gpu_memory

    shutdown_db()
    cleanup_gpu_memory()


# ========================================================================
# Wire all service modules for @inject pattern
# ========================================================================
SERVICE_MODULES = [
    "src.service.preprocessorService",
    "src.service.ingestionService",
    "src.service.verificationService",
    "src.service.hierarchicalMatchingService",
    "src.processor.hierarchicalMatcher",
]


def wire_services() -> None:
    """Wire all service modules with the container for @inject pattern."""
    container.wire(modules=SERVICE_MODULES)


def unwire_services() -> None:
    """Unwire all service modules (for testing)."""
    container.unwire(modules=SERVICE_MODULES)


# ========================================================================
# Helper functions for accessing container providers (backward compatibility)
# ========================================================================

def get_app_config():
    """Get application configuration provider."""
    return container.app_config


def get_sqlalchemy_repo():
    """Get SQLAlchemy repository provider."""
    return container.sqlalchemy_repo


def get_minio_repo():
    """Get MinIO repository provider."""
    return container.minio_repo


def get_milvus_repo():
    """Get Milvus repository provider."""
    return container.milvus_repo


def get_dino_processor():
    """Get DINO processor provider."""
    return container.dino_processor


def get_superpoint_processor():
    """Get SuperPoint processor provider."""
    return container.superpoint_processor


def get_lightglue_processor():
    """Get LightGlue processor provider."""
    return container.lightglue_processor


def get_preprocessor_service():
    """Get preprocessor service provider."""
    return container.preprocessor_service


def get_ingestion_service():
    """Get ingestion service provider."""
    return container.ingestion_service


def get_verification_service():
    """Get verification service provider."""
    return container.verification_service


def get_hierarchical_matching_service():
    """Get hierarchical matching service provider."""
    return container.hierarchical_matching_service


def get_identification_service():
    """Get identification service provider."""
    return container.identification_service
