#!/usr/bin/env python3
"""
Connection Manager - Singleton Pattern for Database Connections

Provides centralized connection management for Milvus and MinIO:
- Singleton pattern to reuse connections
- Connection pooling
- Health checks
- Graceful shutdown
"""

import logging
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from functools import lru_cache
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for connection pool."""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    last_error: Optional[str] = None


class ConnectionManager:
    """Singleton connection manager for Milvus and MinIO.
    
    This class ensures:
    - Single connection instance per repository type
    - Thread-safe initialization
    - Connection health monitoring
    - Graceful cleanup on shutdown
    """
    
    _instance: Optional['ConnectionManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ConnectionManager':
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize connection manager."""
        if self._initialized:
            return
            
        self._initialized = True
        self._milvus_client: Optional[Any] = None
        self._minio_client: Optional[Any] = None
        
        # Connection state
        self._milvus_connected = False
        self._minio_connected = False
        
        # Thread safety
        self._milvus_lock = threading.Lock()
        self._minio_lock = threading.Lock()
        
        # Stats
        self._stats = {
            'milvus': ConnectionStats(),
            'minio': ConnectionStats()
        }
        
        # Configuration
        self._config: Dict[str, Any] = {}
    
    def configure(self, **kwargs) -> None:
        """Configure connection manager with settings.
        
        Args:
            **kwargs: Configuration options
                - milvus_uri: Milvus connection URI
                - milvus_pool_size: Connection pool size
                - minio_endpoint: MinIO endpoint
                - minio_access_key: MinIO access key
                - minio_secret_key: MinIO secret key
                - minio_bucket: MinIO bucket name
        """
        self._config.update(kwargs)
        logger.info("ConnectionManager configured")
    
    @property
    def milvus_client(self) -> Optional[Any]:
        """Get Milvus client (lazy initialization)."""
        return self._milvus_client
    
    @property
    def minio_client(self) -> Optional[Any]:
        """Get MinIO client (lazy initialization)."""
        return self._minio_client
    
    @property
    def milvus_connected(self) -> bool:
        """Check if Milvus is connected."""
        return self._milvus_connected
    
    @property
    def minio_connected(self) -> bool:
        """Check if MinIO is connected."""
        return self._minio_connected
    
    def connect_milvus(self, uri: Optional[str] = None, timeout: float = 30.0) -> bool:
        """Connect to Milvus with singleton pattern.
        
        Args:
            uri: Milvus connection URI
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully
        """
        with self._milvus_lock:
            if self._milvus_client is not None and self._milvus_connected:
                logger.debug("Milvus client already connected")
                return True
            
            try:
                from pymilvus import MilvusClient
                
                uri = uri or self._config.get('milvus_uri', 'http://localhost:19530')
                self._milvus_client = MilvusClient(uri=uri, timeout=timeout)
                
                # Test connection
                self._milvus_client.list_collections()
                
                self._milvus_connected = True
                self._stats['milvus'].total_connections += 1
                self._stats['milvus'].active_connections = 1
                
                logger.info(f"Connected to Milvus: {uri}")
                return True
                
            except Exception as e:
                self._stats['milvus'].failed_connections += 1
                self._stats['milvus'].last_error = str(e)
                logger.error(f"Failed to connect to Milvus: {e}")
                self._milvus_connected = False
                return False
    
    def connect_minio(
        self, 
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: str = "tree-features",
        secure: bool = False
    ) -> bool:
        """Connect to MinIO with singleton pattern.
        
        Args:
            endpoint: MinIO endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket: MinIO bucket name
            secure: Use HTTPS
            
        Returns:
            True if connected successfully
        """
        with self._minio_lock:
            if self._minio_client is not None and self._minio_connected:
                logger.debug("MinIO client already connected")
                return True
            
            try:
                from minio import Minio

                endpoint = endpoint or self._config.get('minio_endpoint', 'localhost:9000')
                access_key = access_key or self._config.get('minio_access_key', '')
                secret_key = secret_key or self._config.get('minio_secret_key', '')

                if not access_key or not secret_key:
                    raise ValueError(
                        "MinIO credentials required: provide access_key/secret_key arguments "
                        "or set minio_access_key/minio_secret_key via configure()"
                    )
                
                self._minio_client = Minio(
                    endpoint,
                    access_key=access_key,
                    secret_key=secret_key,
                    secure=secure
                )
                
                # Check bucket exists
                if not self._minio_client.bucket_exists(bucket):
                    self._minio_client.make_bucket(bucket)
                    logger.info(f"Created MinIO bucket: {bucket}")
                
                self._minio_connected = True
                self._stats['minio'].total_connections += 1
                self._stats['minio'].active_connections = 1
                
                logger.info(f"Connected to MinIO: {endpoint}")
                return True
                
            except Exception as e:
                self._stats['minio'].failed_connections += 1
                self._stats['minio'].last_error = str(e)
                logger.error(f"Failed to connect to MinIO: {e}")
                self._minio_connected = False
                return False
    
    def disconnect_milvus(self) -> None:
        """Disconnect Milvus client."""
        with self._milvus_lock:
            self._milvus_client = None
            self._milvus_connected = False
            self._stats['milvus'].active_connections = 0
            logger.info("Disconnected from Milvus")

    def disconnect_minio(self) -> None:
        """Disconnect MinIO client."""
        with self._minio_lock:
            self._minio_client = None
            self._minio_connected = False
            self._stats['minio'].active_connections = 0
            logger.info("Disconnected from MinIO")
    
    def disconnect_all(self) -> None:
        """Disconnect all clients."""
        self.disconnect_milvus()
        self.disconnect_minio()
        logger.info("Disconnected all database clients")
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all connections.
        
        Returns:
            Dictionary with health status
        """
        health = {
            'milvus': {'connected': self._milvus_connected},
            'minio': {'connected': self._minio_connected}
        }
        
        # Try to verify connections
        if self._milvus_connected and self._milvus_client:
            try:
                self._milvus_client.list_collections()
                health['milvus']['status'] = 'healthy'
            except Exception as e:
                health['milvus']['status'] = 'unhealthy'
                health['milvus']['error'] = str(e)
        
        if self._minio_connected and self._minio_client:
            try:
                # Simple health check - just verify bucket list works
                list(self._minio_client.list_buckets())
                health['minio']['status'] = 'healthy'
            except Exception as e:
                health['minio']['status'] = 'unhealthy'
                health['minio']['error'] = str(e)
        
        return health
    
    def get_stats(self) -> Dict[str, ConnectionStats]:
        """Get connection statistics."""
        return self._stats.copy()
    
    def reset(self) -> None:
        """Reset connection manager (for testing).

        Thread-safe reset that acquires locks sequentially (not nested) to avoid
        ABBA-style deadlocks with any other code that might hold one lock while
        waiting for the other.
        """
        with self._milvus_lock:
            self._milvus_client = None
            self._milvus_connected = False
            self._stats['milvus'].active_connections = 0

        with self._minio_lock:
            self._minio_client = None
            self._minio_connected = False
            self._stats['minio'].active_connections = 0

        self._stats = {
            'milvus': ConnectionStats(),
            'minio': ConnectionStats()
        }
        logger.info("ConnectionManager reset")


@lru_cache(maxsize=1)
def get_connection_manager() -> ConnectionManager:
    """Get cached ConnectionManager instance.
    
    Returns:
        Singleton ConnectionManager instance
    """
    return ConnectionManager()


def reset_connection_manager() -> None:
    """Reset connection manager (for testing)."""
    manager = get_connection_manager()
    manager.reset()
    # Clear both the lru_cache AND the class-level singleton so the next
    # call to ConnectionManager() constructs a truly fresh instance.
    get_connection_manager.cache_clear()
    with ConnectionManager._lock:
        ConnectionManager._instance = None


@contextmanager
def managed_connection(service: str):
    """Context manager for database connections.
    
    Usage:
        with managed_connection('milvus'):
            # do something with Milvus
            pass
    
    Args:
        service: 'milvus' or 'minio'
    """
    manager = get_connection_manager()
    
    if service == 'milvus':
        if not manager.milvus_connected:
            manager.connect_milvus()
        yield manager.milvus_client
    elif service == 'minio':
        if not manager.minio_connected:
            manager.connect_minio()
        yield manager.minio_client
    else:
        raise ValueError(f"Unknown service: {service}")

