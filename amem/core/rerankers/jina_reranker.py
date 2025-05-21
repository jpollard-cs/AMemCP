#!/usr/bin/env python3
"""
Jina Reranker Implementation.
"""

import asyncio
from typing import Dict, List, Optional, Tuple

from sentence_transformers.cross_encoder import CrossEncoder

from amem.core.interfaces import IReranker, SearchResult
from amem.utils.utils import setup_logger

logger = setup_logger(__name__)


class JinaReranker(IReranker):
    """Reranker implementation using Jina AI CrossEncoder models."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        """Initialize the Jina reranker.

        Args:
            model_name: Name of the Jina CrossEncoder model (e.g., 'jina-reranker-v1-turbo-en').
            device: Device to run the model on ('cpu', 'cuda', etc.). Auto-detected if None.
        """
        self.model_name = model_name
        self.device = device
        self._model: Optional[CrossEncoder] = None
        self._initialized = False
        logger.info(f"Initializing JinaReranker with model: {model_name}")
        # Initialize synchronously for now, consider async init if model loading is slow
        self._initialize_model()

    def _initialize_model(self):
        """Load the CrossEncoder model."""
        if self._initialized:
            return
        try:
            # Specify device and trust remote code
            # Use auto dtype for best performance/precision trade-off
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,  # Standard max length for many cross-encoders
                device=self.device,  # Auto-detect if None
                model_kwargs={"torch_dtype": "auto"},
                trust_remote_code=True,  # Often required for Jina models
            )
            self._initialized = True
            logger.info(
                f"Successfully initialized Jina reranker model '{self.model_name}' on device '{self._model.device}'."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Jina reranker model '{self.model_name}': {e}", exc_info=True)
            self._model = None
            self._initialized = False
            # Decide if we should raise here or allow failing gracefully
            # raise # Uncomment to make initialization failure fatal

    async def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank the provided search results based on the query."""
        if not self._initialized or not self._model:
            logger.warning(f"Jina reranker '{self.model_name}' not initialized. Returning original ranking.")
            return results
        if not results:
            return []

        logger.debug(f"Reranking {len(results)} results for query: '{query[:50]}...'")

        # Create query-document pairs for the model
        # Ensure content is available in the SearchResult objects
        pairs: List[Tuple[str, str]] = []
        valid_results_map: Dict[int, SearchResult] = {}
        for _, res in enumerate(results):
            if res.content:
                pairs.append((query, res.content))
                valid_results_map[len(pairs) - 1] = res  # Map pair index back to result
            else:
                logger.warning(f"SearchResult {res.note_id} missing content, cannot rerank.")

        if not pairs:
            logger.warning("No valid query-content pairs found for reranking.")
            return results  # Return original if no content available

        try:
            # Model prediction is CPU/GPU bound, run in executor
            loop = asyncio.get_running_loop()
            scores = await loop.run_in_executor(
                None,  # Default thread pool executor
                self._model.predict,
                pairs,
                True,  # Convert to list for compatibility
            )

            # Combine original results with new scores
            scored_results = []
            for i, score in enumerate(scores):
                original_result = valid_results_map[i]
                # Create a new SearchResult or update the score
                scored_results.append(
                    SearchResult(
                        note_id=original_result.note_id,
                        score=float(score),  # Use the reranker's score
                        content=original_result.content,
                    )
                )

            # Add back any results that couldn't be reranked (e.g., missing content)
            # Decide on placement: append at the end or try to interleave based on original score?
            # Appending is simpler.
            reranked_ids = {res.note_id for res in scored_results}
            for res in results:
                if res.note_id not in reranked_ids:
                    logger.debug(f"Appending non-reranked result {res.note_id} to the end.")
                    # Assign a low score or keep original? Let's assign a very low score.
                    res.score = -1e6  # Assign a very low score
                    scored_results.append(res)

            # Sort by the new reranker scores (descending)
            scored_results.sort(key=lambda x: x.score, reverse=True)

            logger.info(f"Reranking completed. Final result count: {len(scored_results)}.")
            return scored_results

        except Exception as e:
            logger.error(f"Error during Jina reranking: {e}", exc_info=True)
            # Fallback to original ranking on error
            return results
