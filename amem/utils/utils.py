#!/usr/bin/env python3
"""
Shared utilities for AMemCP codebase.
Contains common functionality used across multiple modules.
"""

import logging
import os
from collections import defaultdict
from typing import Optional

from chromadb.config import Settings

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

    # Remove all existing handlers to prevent duplicate logging
    while logger.handlers:
        logger.removeHandler(logger.handlers[0])

    # Prevent propagation to the root logger to avoid duplicate logs
    logger.propagate = False

    # Configure with our custom handler
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


def get_chroma_settings(persist_directory: Optional[str] = None) -> Settings:
    """Creates a ChromaDB Settings object, prioritizing environment variables.

    Args:
        persist_directory: The root directory for ChromaDB persistence (if needed by client).
                           For AsyncHttpClient, this is NOT directly used by the client itself,
                           but might be used by the server it connects to.

    Returns:
        A chromadb.config.Settings object.
    """
    settings_dict = defaultdict(lambda: None)

    # --- Client/Server Settings (for HttpClient/AsyncHttpClient) ---
    # These might be needed if connecting to a remote or specifically configured server
    settings_dict["chroma_server_host"] = os.environ.get("CHROMA_SERVER_HOST", "localhost")
    settings_dict["chroma_server_http_port"] = int(os.environ.get("CHROMA_SERVER_HTTP_PORT", 8001))
    # Typically default to REST API for HttpClients
    # settings_dict["chroma_api_impl"] = os.environ.get("CHROMA_API_IMPL", "rest")

    # --- Persistence Settings (Primarily for the SERVER if using PersistentClient there) ---
    # Although AsyncHttpClient doesn't use this directly, the SERVER it connects to might.
    # If persist_directory is provided, include it.
    if persist_directory:
        settings_dict["persist_directory"] = os.path.abspath(persist_directory)
    else:
        # Default persist directory if not specified
        settings_dict["persist_directory"] = os.environ.get(
            "CHROMA_DB_DIR", os.path.abspath(os.path.join(os.getcwd(), "data", "chroma_db"))
        )
    # Ensure is_persistent reflects whether a directory is set
    settings_dict["is_persistent"] = bool(settings_dict.get("persist_directory"))

    # Telemetry
    settings_dict["anonymized_telemetry"] = os.environ.get("CHROMA_ANONYMIZED_TELEMETRY", "True").lower() == "true"

    # Resetting (allow for development)
    settings_dict["allow_reset"] = os.environ.get("CHROMA_ALLOW_RESET", "True").lower() == "true"

    # Migrations (usually only relevant for persistent setups)
    settings_dict["migrations"] = os.environ.get("CHROMA_MIGRATIONS", "apply")
    settings_dict["migrations_hash_algorithm"] = os.environ.get("CHROMA_MIGRATIONS_HASH_ALGORITHM", "sha256")

    # Filter out None values before creating Settings object
    filtered_settings = {k: v for k, v in settings_dict.items() if v is not None}

    # Create Settings object with all configured parameters
    settings = Settings(**filtered_settings)

    return settings


# Common constants
DEFAULT_PERSIST_DIR = "./data/chroma_db"
DEFAULT_COLLECTION_PREFIX = "amem"

# Default model names
DEFAULT_GEMINI_LLM_MODEL = "gemini-2.5-flash-preview-05-20"
DEFAULT_GEMINI_EMBED_MODEL = "gemini-embedding-exp-03-07"
DEFAULT_JINA_RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
