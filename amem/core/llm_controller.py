"""
LLM Controller Module
Handles interactions with various LLM backends and embedding generation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Import the interface
from amem.core.interfaces import ICompletionProvider, IEmbeddingProvider
from amem.core.llm_providers import get_provider
from amem.utils.utils import setup_logger

# Configure logging
logger = setup_logger(__name__)


# TODO: I'm not sure we need any of this - why not go directly to the provider or maybe we need some different abstraction, but this just seems to be arbitrarily adding two things together
# that should be in different places - like get_empty_values seems like it should be in the MemoryNote cdomain model class
# we can handle this as part of the refactor to separate embedding and completion providers
class BaseLLMController(ABC):
    """Base class for LLM Controllers"""

    @abstractmethod
    def get_empty_values(self, current_time: str) -> Dict[str, Any]:
        """Get empty values for analysis"""


# Modify LLMController to implement IEmbeddingProvider
class LLMController(BaseLLMController, IEmbeddingProvider, ICompletionProvider):
    """Controller for LLM completions and embeddings."""

    def __init__(
        self, backend: str = "openai", model: str = None, embed_model: str = None, api_key: str = None, **kwargs
    ):
        """Initialize LLM controller"""
        # Get the appropriate provider
        self.provider = get_provider(backend=backend, model=model, embed_model=embed_model, api_key=api_key, **kwargs)

        # Store provider details for reference
        self.backend = backend
        self.model = self.provider.model
        self.embed_model = self.provider.embed_model

        logger.info(
            f"Initialized LLM Controller with provider: {backend}, model: {self.model}, embed_model: {self.embed_model}"
        )

    def get_empty_values(self, current_time: str) -> Dict[str, Any]:
        """Get empty values for the note metadata"""
        # This might be better placed elsewhere (e.g., in MemoryNote defaults or a factory)
        # but kept here for now during refactoring.
        return {
            "keywords": [],
            "context": "",
            "summary": "",
            "type": "general",
            "created_at": current_time,
            "updated_at": current_time,
            "related_notes": [],
            "sentiment": "neutral",
            "importance": 0.5,
            "tags": [],
        }

    async def get_completion(self, prompt: str, config: Optional[Dict] = None, **kwargs) -> str:
        """Get completion from the provider (async)"""
        try:
            # Pass the config dict to the provider's method
            return await self.provider.get_completion(prompt=prompt, config=config)
        except Exception as e:
            logger.error(f"Error getting completion from provider: {e}")
            raise

    # Make get_embeddings async and match the IEmbedder interface
    async def get_embeddings(self, text: str, task_type: Optional[str] = None) -> List[float]:
        """Get embeddings from the provider.

        Args:
            text: The text to embed
            task_type: Optional task type for specialized embeddings (specific to provider)

        Returns:
            List of floats representing the embedding vector
        """
        try:
            return await self.provider.get_embeddings(text=text, task_type=task_type)
        except Exception as e:
            logger.error(f"Error getting embeddings from provider: {e}")
            raise
