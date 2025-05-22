"""
LLM Controller Module
Handles interactions with various LLM backends and embedding generation.

Note: This module provides a compatibility layer over the provider classes.
Future versions may use providers directly for better modularity.
"""

from typing import Dict, List, Optional

# Import the interface
from amem.core.interfaces import ICompletionProvider, IEmbeddingProvider
from amem.core.llm_providers import get_provider
from amem.utils.utils import setup_logger

# Configure logging
logger = setup_logger(__name__)


class LLMController(IEmbeddingProvider, ICompletionProvider):
    """
    Controller for LLM completions and embeddings.

    This class provides a unified interface over different LLM providers.
    It's essentially a compatibility wrapper around the provider classes.
    """

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
