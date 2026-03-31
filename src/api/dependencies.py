#!/usr/bin/env python3
"""
FastAPI Dependency Injection Module

This module provides FastAPI dependency injection using the dependency_injector
library's Provide pattern. All dependencies are accessed via Depends(Provide[Container.XXX]).

Usage in FastAPI routes:
    from src.api.dependencies import get_ingestion_service, get_verification_service

    @router.post("/ingest")
    async def ingest(
        image: UploadFile,
        service: IngestionService = Depends(get_ingestion_service),
    ):
        return await service.ingest(image)

This module should be used in FastAPI routes, while the main container.py
should only be used for application initialization and lifespan management.
"""

from dependency_injector.wiring import Provide, inject
from fastapi import Depends
from typing import Optional, Any

# Import container
from src.config.containers import Container, container

# =============================================================================
# Service Dependencies - Use these in FastAPI route handlers
# =============================================================================
# These functions return the provider for each service, allowing FastAPI
# to inject the correct service instance when the route is called.

def get_ingestion_service():
    """Get IngestionService via DI.
    
    Usage:
        @router.post("/ingest")
        async def ingest(
            service: IngestionService = Depends(get_ingestion_service)
        ):
            return service.ingest(...)
    """
    return Depends(Provide[Container.ingestion_service])


def get_verification_service():
    """Get VerificationService via DI.
    
    Usage:
        @router.post("/verify")
        async def verify(
            image: UploadFile,
            service: VerificationService = Depends(get_verification_service)
        ):
            return await service.verify(...)
    """
    return Depends(Provide[Container.verification_service])


def get_hierarchical_matching_service():
    """Get HierarchicalMatchingService via DI.
    
    Usage:
        @router.post("/match")
        async def match(
            service: HierarchicalMatchingService = Depends(get_hierarchical_matching_service)
        ):
            return await service.match(...)
    """
    return Depends(Provide[Container.hierarchical_matching_service])


def get_identification_service():
    """Get IdentificationService via DI.
    
    Usage:
        @router.get("/identify")
        async def identify(
            service: IdentificationService = Depends(get_identification_service)
        ):
            return service.identify(...)
    """
    return Depends(Provide[Container.identification_service])


# =============================================================================
# Preprocessor Dependencies
# =============================================================================

def get_preprocessor_service():
    """Get PreprocessorService via DI.
    
    Usage:
        @router.post("/preprocess")
        async def preprocess(
            image: UploadFile,
            preprocessor: PreprocessorService = Depends(get_preprocessor_service)
        ):
            return preprocessor.process(image)
    """
    return Depends(Provide[Container.preprocessor_service])


# =============================================================================
# Processor Dependencies - For fine-grained control
# =============================================================================

def get_dino_processor():
    """Get DinoProcessor via DI.
    
    Usage:
        @router.post("/extract-dino")
        async def extract_dino(
            image: UploadFile,
            processor: DinoProcessor = Depends(get_dino_processor)
        ):
            return processor.extract(image)
    """
    return Depends(Provide[Container.dino_processor])


def get_superpoint_processor():
    """Get SuperPointProcessor via DI.
    
    Usage:
        @router.post("/extract-superpoint")
        async def extract_superpoint(
            image: UploadFile,
            processor: SuperPointProcessor = Depends(get_superpoint_processor)
        ):
            return processor.extract(image)
    """
    return Depends(Provide[Container.superpoint_processor])


def get_lightglue_processor():
    """Get LightGlueProcessor via DI.
    
    Usage:
        @router.post("/match")
        async def match(
            query_features,
            candidate_features,
            processor: LightGlueProcessor = Depends(get_lightglue_processor)
        ):
            return processor.match(query_features, candidate_features)
    """
    return Depends(Provide[Container.lightglue_processor])


# =============================================================================
# Repository Dependencies - For advanced use cases
# =============================================================================

def get_sqlalchemy_repo():
    """Get SQLAlchemyORMRepository via DI.
    
    Usage:
        @router.get("/evidence/{id}")
        async def get_evidence(
            evidence_id: str,
            repo: SQLAlchemyORMRepository = Depends(get_sqlalchemy_repo)
        ):
            return repo.get_evidence(evidence_id)
    """
    return Depends(Provide[Container.sqlalchemy_repo])


def get_minio_repo():
    """Get MinIORepository via DI.
    
    Usage:
        @router.get("/features/{key}")
        async def get_features(
            key: str,
            repo: MinIORepository = Depends(get_minio_repo)
        ):
            return repo.load_features(key)
    """
    return Depends(Provide[Container.minio_repo])


def get_milvus_repo():
    """Get MilvusRepository via DI.
    
    Usage:
        @router.post("/search")
        async def search_vectors(
            vector: List[float],
            repo: MilvusRepository = Depends(get_milvus_repo)
        ):
            return repo.search(query_vector=vector, top_k=10)
    """
    return Depends(Provide[Container.milvus_repo])


# =============================================================================
# Configuration Dependency
# =============================================================================

def get_app_config():
    """Get AppConfig via DI.
    
    Usage:
        @router.get("/config")
        async def get_config(
            config: AppConfig = Depends(get_app_config)
        ):
            return {"dino_device": config.dino_device, ...}
    """
    return Depends(Provide[Container.app_config])


# =============================================================================
# Injected Service Classes (Recommended Pattern)
# =============================================================================
# For cleaner FastAPI route handlers, use the @inject decorator pattern
# or type annotations directly with Depends(Provide[...])

# Type aliases for convenience (these are just Provide objects)
IngestionServiceDep = Provide[Container.ingestion_service]
VerificationServiceDep = Provide[Container.verification_service]
HierarchicalMatchingServiceDep = Provide[Container.hierarchical_matching_service]
IdentificationServiceDep = Provide[Container.identification_service]
PreprocessorServiceDep = Provide[Container.preprocessor_service]
DinoProcessorDep = Provide[Container.dino_processor]
SuperPointProcessorDep = Provide[Container.superpoint_processor]
LightGlueProcessorDep = Provide[Container.lightglue_processor]
SQLAlchemyRepoDep = Provide[Container.sqlalchemy_repo]
MinIORepoDep = Provide[Container.minio_repo]
MilvusRepoDep = Provide[Container.milvus_repo]
AppConfigDep = Provide[Container.app_config]


# =============================================================================
# Example FastAPI Route Patterns
# =============================================================================

"""
Example 1: Basic Service Injection

from fastapi import APIRouter, Depends, UploadFile, File
from src.api.dependencies import get_ingestion_service
from src.service import IngestionService

router = APIRouter()

@router.post("/ingest")
async def ingest_tree_image(
    image: UploadFile = File(...),
    service: IngestionService = Depends(get_ingestion_service)
):
    # Read image and call service
    contents = await image.read()
    # ... process image ...
    result = service.ingest(image, mask, image_id, tree_id, metadata)
    return result


Example 2: Multiple Service Dependencies

from src.api.dependencies import get_ingestion_service, get_verification_service
from src.service import IngestionService, VerificationService

@router.post("/ingest-and-verify")
async def ingest_and_verify(
    image: UploadFile = File(...),
    ingest_service: IngestionService = Depends(get_ingestion_service),
    verify_service: VerificationService = Depends(get_verification_service)
):
    # Ingest first
    ingest_result = ingest_service.ingest(...)
    # Then verify
    verify_result = await verify_service.verify(...)
    return {"ingest": ingest_result, "verify": verify_result}


Example 3: Using Provide directly with type annotation

from dependency_injector.wiring import Provide
from fastapi import Depends
from containers import Container

@router.get("/health")
async def health_check(
    config = Depends(Provide[Container.app_config])
):
    return {"status": "healthy", "verbose": config.verbose}


Example 4: Using @inject decorator

from dependency_injector.wiring import inject
from fastapi import APIRouter

router = APIRouter()

@router.get("/stats")
@inject
async def get_stats(
    ingestion_service: IngestionService = Depends(Provide[Container.ingestion_service]),
    verification_service: VerificationService = Depends(Provide[Container.verification_service])
):
    return {
        "ingestion_count": ingestion_service.get_count(),
        "verification_count": verification_service.get_count()
    }
"""


# =============================================================================
# Wiring Check Function
# =============================================================================

def verify_wiring() -> bool:
    """Verify that all dependencies are properly wired.
    
    This function should be called during application startup to ensure
    all dependencies are correctly configured.
    
    Returns:
        True if all dependencies are properly wired, False otherwise
    """
    try:
        # Check that all service providers exist
        providers_to_check = [
            Container.ingestion_service,
            Container.verification_service,
            Container.hierarchical_matching_service,
            Container.identification_service,
            Container.preprocessor_service,
            Container.dino_processor,
            Container.superpoint_processor,
            Container.lightglue_processor,
            Container.sqlalchemy_repo,
            Container.minio_repo,
            Container.milvus_repo,
            Container.app_config,
        ]
        
        for provider in providers_to_check:
            if provider is None:
                return False
        
        return True
    except Exception as e:
        print(f"Error verifying wiring: {e}")
        return False
