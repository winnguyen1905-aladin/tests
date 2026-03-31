# Unified config module exports
# Import from config.py for lazy loading (avoids torch dependency at import time)

from .config import (
    # Logging
    setup_logging,
    # Settings
    Settings,
    get_settings,
    reload_settings,
    # AppConfig bridge
    init_config,
    # Re-export from appConfig (lazy)
    get_config,
)

# Lazy imports to avoid torch dependency at module load time
# These are imported on-demand via __getattr__
_appconfig_exports = {
    "DetectionConfig",
    "DetectionMode",
    "AppConfig",
    "set_config",
    "load_config",
    "load_from_env",
}


def __getattr__(name: str):
    """Lazy import for appConfig classes to avoid torch dependency."""
    if name in _appconfig_exports:
        from .appConfig import (
            DetectionConfig,
            DetectionMode,
            AppConfig,
            set_config,
            load_config,
            load_from_env,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Logging
    "setup_logging",
    # Settings
    "Settings",
    "get_settings",
    "reload_settings",
    # AppConfig bridge
    "init_config",
    "get_config",
    # Lazy imports (appConfig)
    "AppConfig",
    "DetectionConfig",
    "DetectionMode",
    "set_config",
    "load_config",
    "load_from_env",
]
    