#!/usr/bin/env python3
"""
AppConfig - Application Configuration

Centralized configuration management for the SAM3 tree identification system.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
import torch


@dataclass
class AppConfig:
    """Main application configuration."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    model_path: str = "sam3.pt"
    data_dir: Path = field(default_factory=lambda: Path("data"))

    # SAM3 Settings - Optimized for outdoor durian tree detection
    sam3_device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    sam3_confidence: float = 0.05  # Balanced for outdoor scenes
    sam3_image_size: int = 644  # Must be multiple of SAM3/ViT-H stride 14 (640 → 644)
    sam3_text_prompts: List[str] = field(default_factory=lambda: [
        "the entire large tree in the center, including the trunk, branches, and leaves"
    ])

    # Performance settings
    use_cuda_streams: bool = True  # Enable CUDA streams when CUDA is available
    enable_torch_compile: bool = True  # Use torch.compile for faster inference (PyTorch 2.0+)
    use_half_precision: bool = True  # Use FP16 for faster inference on modern GPUs

    # Model Mode: 'ultra' (maximum performance) or 'lite' (memory efficient)
    model_mode: str = "ultra"  # Set via pydantic-settings (MODEL_MODE env)

    # DINO Settings - Optimized for RTX 5060 Ti with dinov3-vitb16-pretrain-lvd1689m (384 dims, ViT-Base 21M)
    dino_model_type: str = "dinov3-vitb16-pretrain-lvd1689m"

    dino_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dino_image_size: int = 644  # Match SAM3 for consistency
    dino_hf_token: str = ""
    dino_use_multi_gpu: bool = False  # Single GPU sufficient
    dino_gpu_ids: Optional[list] = None
    dino_enable_memory_optimization: bool = True
    dino_use_gradient_checkpointing: bool = False  # Not needed for inference
    dino_use_4bit_quantization: bool = False  # Full precision

    # SuperPoint Settings - High keypoints for outdoor scenes
    sp_max_keypoints: int = 4096
    sp_max_dimension: int = 1280 
    sp_device: str = "cuda"  # Force CUDA for maximum performance

    # LightGlue Settings - Optimized matching
    lg_device: str = "cuda"  # Force CUDA for maximum performance
    lg_filter_threshold: float = 0.5
    lg_confidence: float = 0.1

    # Vector Store Settings (set via pydantic-settings)
    vector_store_type: str = "postgres"

    # Milvus Settings (set via pydantic-settings)
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "tree_features"
    milvus_vector_dim: int = 384  # dinov3-vitb16-pretrain-lvd1689m: ViT-Base 21M, 384 dims
    milvus_top_k: int = 100  # Retrieve more candidates for better accuracy
    milvus_search_top_k: int = 100  # For coarse retrieval stage
    milvus_search_nprobe: int = 32  # Number of probes for IVF search

    # PostgreSQL Settings (set via pydantic-settings)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "sam3"
    postgres_user: str = "sam3user"
    postgres_password: str = "sam3pass"
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
    postgres_vector_dim: int = 384
    postgres_vector_index_type: str = "hnsw"
    postgres_vector_m: int = 16
    postgres_vector_ef_construction: int = 64

    # MinIO Settings (set via pydantic-settings)
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "tree-features"
    minio_secure: bool = False

    # Matching Strategy - Optimized for outdoor durian tree identification
    coarse_threshold: float = 0.5
    inlier_threshold: int = 8

    # Hierarchical Matching Thresholds
    # min_matches: Minimum number of LightGlue matched keypoints required
    # min_inliers: Minimum number of RANSAC inliers required
    min_matches: int = 15
    min_inliers: int = 7

    # Filter Ranges (for temporary disabling, set to large values)
    # These are used in main.py for geo_filter and angle_filter
    geo_radius_meters: float = 10.0  # 50km radius (disable by setting large)
    hor_angle_range: float = 30.0  # ±180 degrees (almost full range)
    ver_angle_range: float = 60.0   # ±90 degrees (full vertical range)
    pitch_range: float = 75.0       # ±90 degrees (full pitch range)

    # General
    verbose: bool = True
    log_level: str = "DEBUG"

    # Multi-signal scoring weights - Optimized for durian tree matching
    feature_weights: dict = field(default_factory=lambda: {
        "dino_global": 0.70,
        "keypoint_density": 0.15,
        "feature_matching": 0.15,
        "scale_consistency": 0.0,
    })

    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported DINO models."""
        return [
            "dinov2-giant",
            "dinov2-vitl14",
            "dinov2-vitb14",
            "dinov3-vitb16-pretrain-lvd1689m",
            "dinov2-vits14",
            "dinov3-vitl16",
            "dinov3-vitg14",
            "dinov3-convnext-small-pretrain-lvd1689m",
        ]

    def load_from_file(self, config_path: str) -> 'AppConfig':
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to config file
        
        Returns:
            Updated AppConfig instance
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        return self
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            config_path: Path to config file
        """
        data = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_repository_config(self) -> dict:
        """Get configuration for repositories."""
        return {
            "vector_store_type": self.vector_store_type,
            "milvus": {
                "uri": self.milvus_uri,
                "collection": self.milvus_collection,
                "vector_dim": self.milvus_vector_dim,
            },
            "postgresql": {
                "host": self.postgres_host,
                "port": self.postgres_port,
                "database": self.postgres_db,
                "user": self.postgres_user,
                "password": self.postgres_password,
                "pool_size": self.postgres_pool_size,
                "max_overflow": self.postgres_max_overflow,
                "vector_dim": self.postgres_vector_dim,
                "vector_index_type": self.postgres_vector_index_type,
            },
            "minio": {
                "endpoint": self.minio_endpoint,
                "access_key": self.minio_access_key,
                "secret_key": self.minio_secret_key,
                "bucket": self.minio_bucket,
                "secure": self.minio_secure,
            }
        }

    def get_processor_config(self, processor_type: str) -> dict:
        """Get configuration for a specific processor.

        Args:
            processor_type: 'sam3', 'dino', 'superpoint', 'lightglue'

        Returns:
            Processor configuration dict
        """
        configs = {
            "sam3": {
                "model_path": self.model_path,
                "device": self.sam3_device,
                "confidence_threshold": self.sam3_confidence,
                "verbose": self.verbose,
            },
            "dino": {
                "model_type": self.dino_model_type,
                "device": self.dino_device,
                "image_size": self.dino_image_size,
                "hf_token": self.dino_hf_token,
                "verbose": self.verbose,
            },
            "superpoint": {
                "max_keypoints": self.sp_max_keypoints,
                "max_dimension": self.sp_max_dimension,
                "device": self.lg_device,
                "verbose": self.verbose,
            },
            "lightglue": {
                "device": self.lg_device,
                "filter_threshold": self.lg_filter_threshold,
                "verbose": self.verbose,
            },
        }

        return configs.get(processor_type, {})

    def get_matching_config(self) -> dict:
        """Get configuration for matching strategy."""
        return {
            "coarse_threshold": self.coarse_threshold,
            "inlier_threshold": self.inlier_threshold,
            "top_k": self.milvus_top_k,
        }

class DetectionMode(Enum):
    """Detection mode enumeration."""
    TEXT = "text"  # Text-based concept segmentation
    EXEMPLAR = "exemplar"  # Image exemplar-based segmentation
    POINT = "point"  # Point-based visual segmentation
    BOX = "box"  # Bounding box-based visual segmentation
    MASK = "mask"  # Mask-based visual segmentation


@dataclass
class DetectionConfig:
    """Configuration for SAM3 detection."""

    # General settings
    mode: DetectionMode
    model_path: str = "sam3.pt"
    device: str = "cuda"  # "cuda" or "cpu"
    half_precision: bool = True  # Use FP16 for faster inference

    # Input/Output
    input_image: str = ""
    output_dir: str = "output"
    save_masks: bool = True
    save_boxes: bool = True
    save_visualization: bool = True

    # Detection parameters
    confidence_threshold: float = 0.08
    mask_threshold: float = 0.08
    refine_masks: bool = False  # Enable morphological refinement
    erosion_kernel_size: int = 3  # Kernel size for erosion (shrink mask)
    use_tracking: bool = False  # Enable video object tracking
    tracker_type: str = "sam3_tracker_quantized"  # Tracker: sam3_tracker, sam3_tracker_quantized
    verbose: bool = True

    # Memory optimization settings
    max_detections: int = 100  # Limit number of detections to save memory
    torch_memory_fraction: float = 0.9  # Fraction of GPU memory to use (0.0-1.0)

    # Text-based detection (mode: TEXT)
    text_prompts: List[str] = field(default_factory=list)

    # Exemplar-based detection (mode: EXEMPLAR)
    exemplar_boxes: List[List[float]] = field(default_factory=list)  # [[x1, y1, x2, y2], ...]

    # Point-based detection (mode: POINT)
    points: List[List[float]] = field(default_factory=list)  # [[x, y], ...]
    point_labels: List[int] = field(default_factory=list)  # [1 for positive, 0 for negative]

    # Box-based detection (mode: BOX)
    bounding_boxes: List[List[float]] = field(default_factory=list)  # [[x1, y1, x2, y2], ...]

    # Mask-based detection (mode: MASK)
    input_masks: Optional[str] = None  # Path to mask file

    resize_input_width: Optional[int] = None # Resize input video/image width (maintain aspect ratio)
    frame_limit: Optional[int] = None # Limit number of frames to process

    # Visualization settings
    show_labels: bool = True
    show_scores: bool = True
    show_boxes: bool = True
    overlay_alpha: float = 0.5
    colormap: str = "rainbow"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Convert mode string to enum if needed
        if isinstance(self.mode, str):
            self.mode = DetectionMode(self.mode.lower())

        # Validate input image exists
        if self.input_image and not Path(self.input_image).exists():
            raise ValueError(f"Input image not found: {self.input_image}")

        # Validate mode-specific parameters
        self._validate_mode_params()

        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _validate_mode_params(self) -> None:
        """Validate mode-specific parameters."""
        if self.mode == DetectionMode.TEXT:
            if not self.text_prompts:
                raise ValueError("Text prompts are required for TEXT mode")

        elif self.mode == DetectionMode.EXEMPLAR:
            if not self.exemplar_boxes:
                raise ValueError("Exemplar boxes are required for EXEMPLAR mode")
            for box in self.exemplar_boxes:
                if len(box) != 4:
                    raise ValueError(f"Exemplar box must have 4 coordinates: {box}")

        elif self.mode == DetectionMode.POINT:
            if not self.points:
                raise ValueError("Points are required for POINT mode")
            if not self.point_labels:
                # Default to all positive points
                self.point_labels = [1] * len(self.points)
            if len(self.points) != len(self.point_labels):
                raise ValueError("Number of points must match number of labels")
            for point in self.points:
                if len(point) != 2:
                    raise ValueError(f"Point must have 2 coordinates: {point}")

        elif self.mode == DetectionMode.BOX:
            if not self.bounding_boxes:
                raise ValueError("Bounding boxes are required for BOX mode")
            for box in self.bounding_boxes:
                if len(box) != 4:
                    raise ValueError(f"Bounding box must have 4 coordinates: {box}")

        elif self.mode == DetectionMode.MASK:
            if not self.input_masks:
                raise ValueError("Input masks are required for MASK mode")
            if not Path(self.input_masks).exists():
                raise ValueError(f"Mask file not found: {self.input_masks}")


# Global configuration instance (populated from pydantic-settings on first access)
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance.

    On first access, builds AppConfig from pydantic-settings (Settings).
    No os.getenv() in this module; all env-backed values come from Settings.
    """
    global _config
    if _config is None:
        from src.config.config import create_app_config_from_settings, get_settings
        _config = create_app_config_from_settings(get_settings())
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config(config_path: str) -> AppConfig:
    """Load configuration from file and set as global."""
    config = AppConfig().load_from_file(config_path)
    set_config(config)
    return config


def get_supported_models() -> List[str]:
    """Get list of supported DINO models."""
    return AppConfig.get_supported_models()


# Environment variable overrides (via pydantic-settings)
def load_from_env() -> AppConfig:
    """Load configuration from environment variables.

    Uses pydantic-settings (Settings) to read env and .env file; builds AppConfig
    from it and sets as global. No direct os.getenv() in this module.
    """
    from src.config.config import create_app_config_from_settings, get_settings
    settings = get_settings()
    config = create_app_config_from_settings(settings)
    set_config(config)
    return config


# if __name__ == "__main__":
#     # Example: Create and save default config
#     config = AppConfig(verbose=True)

#     # Save to file
#     config.save_to_file("sam3_config.json")
#     print("Default config saved to sam3_config.json")

#     # Print config
#     print("\n=== AppConfig ===")
#     print(f"SAM3 Model: {config.model_path}")
#     print(f"DINO Model: {config.dino_model_type}")
#     print(f"SuperPoint Keypoints: {config.sp_max_keypoints}")
#     print(f"MILVUS URI: {config.milvus_uri}")

#     # Print supported models
#     print("\n=== Supported DINO Models ===")
#     for model in get_supported_models():
#         print(f"  - {model}")
#     print(f"DINO Model: {config.dino_model_type}")
#     print(f"Milvus URI: {config.milvus_uri}")
#     print(f"MinIO Endpoint: {config.minio_endpoint}")
#     print(f"Inlier Threshold: {config.inlier_threshold}")
#     print(f"Coarse Threshold: {config.coarse_threshold}")

