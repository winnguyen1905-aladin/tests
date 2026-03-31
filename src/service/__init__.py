"""
Service Layer - Business Logic Orchestration

Services coordinate processors, repositories, and strategies
to implement the ingestion and verification workflows.

Dependency Injection:
    All services use @inject decorator with dependency_injector.
    Wire with: container.wire(modules=["src.service.*"])
    
    Example:
        # Via FastAPI Depends (recommended):
        from src.api.dependencies import get_ingestion_service
        
        @router.post("/ingest")
        async def ingest(service: IngestionService = Depends(get_ingestion_service)):
            ...
        
        # Via container directly:
        from containers import container
        container.wire(modules=["src.service.ingestionService"])
        service = container.ingestion_service()
"""

from .preprocessorService import PreprocessorService
from .ingestionService import IngestionService, IngestionResult
from .verificationService import (
    VerificationService,
    MatchCandidate,
    HierarchicalVerificationService,
)
from .hierarchicalMatchingService import (
    HierarchicalMatchingService,
    HierarchicalVerificationService,
)
__all__ = [
    # Preprocessor
    "PreprocessorService",
    
    # Ingestion
    "IngestionService",
    "IngestionResult",
    
    # Verification
    "VerificationService",
    "MatchCandidate",
    "HierarchicalVerificationService",
    
    # Hierarchical Matching
    "HierarchicalMatchingService",
]
