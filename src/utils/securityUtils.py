"""
Shared security utilities for SAM3.

Provides helpers for:
- Storage identifier sanitization (path traversal prevention)
- Multipart field name sanitization
- Config file path validation
- Query vector validation
"""

import math
import re
import uuid
from pathlib import Path

# Allowlist pattern: alphanumeric, hyphen, underscore, dot — no slashes, no dots-only segments
_SAFE_ID_RE = re.compile(r'^[A-Za-z0-9_\-.]{1,128}$')
_DOTDOT_RE = re.compile(r'(\.\./|/\.\.|^\.\.$|/\.\.$)')

# Absolute project root — paths for config files must stay within this tree
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def sanitize_storage_id(value: str, field: str = "id") -> str:
    """Validate and return a safe storage identifier.

    Accepts identifiers that consist solely of alphanumeric characters, hyphens,
    underscores, and dots, with a maximum length of 128 characters.

    Raises:
        ValueError: if value is empty, contains null bytes, path-traversal sequences,
                    or characters outside the allowlist.
    """
    if not value or not isinstance(value, str):
        raise ValueError(f"{field} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{field} contains null bytes")
    if ".." in value or value.startswith("/") or value.startswith("\\"):
        raise ValueError(f"{field} contains path-traversal characters")
    if not _SAFE_ID_RE.match(value):
        raise ValueError(
            f"{field} contains unsafe characters. "
            "Only alphanumeric, hyphen, underscore, and dot are allowed (max 128 chars)."
        )
    return value


def safe_field_name_as_id(field_name: str) -> str:
    """Convert an untrusted multipart field name to a safe image identifier.

    Strips every character outside the allowlist and truncates to 64 characters.
    Falls back to a random UUID fragment when the result would be empty.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_\-.]", "_", field_name)[:64].strip("_")
    return sanitized if sanitized else f"upload-{uuid.uuid4().hex[:8]}"


def validate_config_path(config_path: str) -> Path:
    """Resolve a config file path and ensure it stays within the project root.

    Raises:
        ValueError: if the resolved path escapes the project root directory.
    """
    resolved = Path(config_path).resolve()
    if not str(resolved).startswith(str(_PROJECT_ROOT) + "/") and resolved != _PROJECT_ROOT:
        raise ValueError(
            f"Config path '{resolved}' is outside the project root '{_PROJECT_ROOT}'. "
            "Refusing to open."
        )
    return resolved


def validate_vector(query_vector, name: str = "query_vector") -> list:
    """Validate every element of a query vector is a finite float.

    Returns a plain list of Python floats.  Raises ValueError if any element
    is non-numeric, NaN, or infinite.
    """
    result: list[float] = []
    for i, v in enumerate(query_vector):
        try:
            f = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{i}] cannot be converted to float: {v!r}") from exc
        if not math.isfinite(f):
            raise ValueError(f"{name}[{i}] is not finite (got {v!r})")
        result.append(f)
    return result
