#!/usr/bin/env python3
"""
Debug routes for SAM3 API.

Contains debug and diagnostic endpoints for troubleshooting and monitoring.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from src.api.lifespan import get_verification_service
from src.config.appConfig import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/features")
async def debug_features() -> Dict[str, Any]:
    """Debug endpoint to check stored features."""
    verification_service = get_verification_service()
    if not verification_service:
        raise HTTPException(status_code=503, detail="Verification service not initialized")

    try:
        # Get feature information from PostgreSQL
        if hasattr(verification_service, 'postgres_repo') and verification_service.postgres_repo:
            try:
                count = verification_service.postgres_repo.get_evidence_count()
                return {
                    "success": True,
                    "total_entities": count,
                    "total_features": count,
                    "features_by_tree": {},
                    "message": "Feature information retrieved from PostgreSQL"
                }
            except Exception as e:
                logger.warning(f"Could not get evidence count: {e}")
                return {
                    "success": True,
                    "total_entities": 0,
                    "total_features": 0,
                    "features_by_tree": {},
                    "message": f"PostgreSQL available but error: {e}"
                }
        else:
            return {
                "success": True,
                "total_entities": 0,
                "total_features": 0,
                "features_by_tree": {},
                "message": "PostgreSQL not available"
            }
    except Exception as e:
        logger.error(f"Debug features error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/postgres")
async def debug_postgres() -> Dict[str, Any]:
    """Debug endpoint to check PostgreSQL directly."""
    try:
        app_cfg = get_config()
        from src.repository.databaseManager import get_db_session
        from src.config.containers import container

        # Get repository from container
        repo = container.sqlalchemy_repo()

        try:
            count = repo.get_evidence_count()
        except Exception as e:
            logger.error(f"Could not get evidence count: {e}")
            return {
                "success": False,
                "error": f"Cannot connect to PostgreSQL: {e}",
                "host": app_cfg.postgres_host,
                "port": app_cfg.postgres_port,
                "database": app_cfg.postgres_db
            }

        return {
            "success": True,
            "database": app_cfg.postgres_db,
            "evidence_count": count,
            "vector_dimension": app_cfg.postgres_vector_dim,
        }
    except Exception as e:
        logger.error(f"Debug PostgreSQL error: {e}")
        raise HTTPException(status_code=500, detail=str(e))