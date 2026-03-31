"""
Processor Package

Feature extraction and matching processors.
"""

from .sam3ImageProcessor import SAM3Detector, DetectionConfig, DetectionMode
from .dinoProcessor import DinoProcessor, DinoConfig, DinoV3Processor
from .superPointProcessor import SuperPointProcessor, SuperPointConfig

# Optional imports
try:
    from .lightglueProcessor import LightGlueProcessor, LightGlueConfig
    _has_lightglue = True
except ImportError:
    _has_lightglue = False

try:
    from .hierarchicalMatcher import (
        HierarchicalMatcher,
        MatchingResult,
        create_hierarchical_matcher
    )
    _has_hierarchical = True
except ImportError:
    _has_hierarchical = False

__all__ = [
    "SAM3Detector",
    "DetectionConfig",
    "DetectionMode",
    "DinoProcessor",
    "DinoConfig",
    "DinoV3Processor",
    "SuperPointProcessor",
    "SuperPointConfig",
    "MatchingResult"
]

if _has_lightglue:
    __all__.extend(["LightGlueProcessor", "LightGlueConfig"])

if _has_hierarchical:
    __all__.extend(["HierarchicalMatcher", "MatchingResult", "create_hierarchical_matcher"])

