#!/usr/bin/env python3
"""
Shared utilities for AMemCP codebase.
Contains common functionality used across multiple modules.
"""

import logging
from typing import Any, Dict, Optional

try:
    import colorlog

    has_colorlog = True
except ImportError:
    has_colorlog = False
    print("For colorful logs, install colorlog: pip install colorlog")


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up pretty, colorful logging.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist (to avoid duplicate configuration)
    if not logger.hasHandlers():
        if has_colorlog:
            # Create a colorlog handler
            handler = colorlog.StreamHandler()
            handler.setFormatter(
                colorlog.ColoredFormatter(
                    "%(log_color)s%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    log_colors={
                        "DEBUG": "cyan",
                        "INFO": "green",
                        "WARNING": "yellow",
                        "ERROR": "red",
                        "CRITICAL": "red,bg_white",
                    },
                    secondary_log_colors={
                        "message": {
                            "DEBUG": "cyan",
                            "INFO": "white",
                            "WARNING": "yellow",
                            "ERROR": "red",
                            "CRITICAL": "red,bg_white",
                        }
                    },
                )
            )
        else:
            # Standard handler without colors
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            )

        # Set the log level
        logger.setLevel(level)

        # Add our custom handler
        logger.addHandler(handler)

    return logger


def get_chroma_settings(persist_directory: Optional[str] = None) -> Dict[str, Any]:
    """Get ChromaDB settings with version-specific configuration.

    Args:
        persist_directory: Directory for persistent storage (optional)

    Returns:
        Dictionary of ChromaDB settings
    """
    from chromadb.config import Settings

    settings_dict = {}

    # Core settings
    if persist_directory:
        settings_dict["persist_directory"] = persist_directory
        settings_dict["is_persistent"] = True

    # Always allow reset for development convenience
    settings_dict["allow_reset"] = True

    # For ChromaDB 1.0.0 compatibility with AsyncHttpClient
    settings_dict["chroma_server_host"] = "localhost"
    settings_dict["chroma_server_http_port"] = 8001  # Default is 8000, using 8001 to avoid conflict

    # Telemetry settings
    # Disable anonymized telemetry for privacy if needed
    settings_dict["anonymized_telemetry"] = False

    # Maintenance settings
    # Apply database migrations automatically
    settings_dict["migrations"] = "apply"
    # Use SHA-256 for migrations hash (more secure than default MD5)
    settings_dict["migrations_hash_algorithm"] = "sha256"

    # Create Settings object with all configured parameters
    return Settings(**settings_dict)


# Common constants
DEFAULT_PERSIST_DIR = "./data/chroma_db"
DEFAULT_COLLECTION_PREFIX = "amem"

# Default model names
DEFAULT_GEMINI_LLM_MODEL = "gemini-1.5-pro-latest"
DEFAULT_GEMINI_EMBED_MODEL = "text-embedding-004"
DEFAULT_JINA_RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
