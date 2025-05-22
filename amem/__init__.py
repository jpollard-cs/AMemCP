"""Agentic Memory System - A memory system for agentic applications."""

__version__ = "0.1.0"

from amem.core import AgenticMemorySystem, LLMController, MemoryNote
from amem.utils import get_chroma_settings, setup_logger

__all__ = [
    "AgenticMemorySystem",
    "MemoryNote",
    "LLMController",
    "setup_logger",
    "get_chroma_settings",
]
