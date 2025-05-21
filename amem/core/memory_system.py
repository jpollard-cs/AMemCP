#!/usr/bin/env python3
"""
Agentic Memory System implementation with LLM integration, retrievers, and reranking.
Uses async I/O throughout for optimal performance.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from amem.core.interfaces import (
    IContentAnalyzer,
    IEmbeddingProvider,
    IManagesOwnIndex,
    IMemoryCache,
    IMemoryStore,
    IReranker,
    IRetriever,
    SearchResult,
)
from amem.core.models.memory_note import MemoryNote
from amem.utils.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class AgenticMemorySystemConfig:
    """Configuration settings for the AgenticMemorySystem."""

    enable_llm_analysis: bool = True
    enable_auto_segmentation: bool = False
    # Add other future configuration options here as needed


class AgenticMemorySystem:
    """Orchestrates memory operations using swappable components."""

    def __init__(
        self,
        memory_store: IMemoryStore,
        cache: IMemoryCache,
        embedder: IEmbeddingProvider,
        analyzer: IContentAnalyzer,
        retriever: IRetriever,
        config: AgenticMemorySystemConfig,
        reranker: Optional[IReranker] = None,
    ):
        """Initialize the Agentic Memory System with dependency injection.

        Args:
            memory_store: Implementation for persistent storage (IMemoryStore).
            cache: Implementation for in-memory caching (IMemoryCache).
            embedder: Implementation for generating embeddings (IEmbedder).
            analyzer: Implementation for content analysis (IContentAnalyzer).
            retriever: Implementation for retrieving memories (IRetriever).
            config: Configuration object for the memory system.
            reranker: Optional implementation for reranking search results (IReranker).
        """
        self.store = memory_store
        self.cache = cache
        self.embedder = embedder
        self.analyzer = analyzer
        self.retriever = retriever
        self.reranker = reranker
        self.config = config

        self._initialized = False
        logger.info("AgenticMemorySystem instance created. Call initialize() to load data.")

    async def initialize(self, load_existing: bool = True) -> None:
        """Initialize the memory system components asynchronously.

        Args:
            load_existing: If True, load existing notes from store into cache and retriever indices.
        """
        if self._initialized:
            logger.debug("Memory system already initialized.")
            return

        logger.info("Initializing Agentic Memory System components...")
        try:
            # Initialize store first (might create resources needed by others)
            if hasattr(self.store, "initialize"):
                logger.info("Initializing Memory Store...")
                await self.store.initialize()

            # Initialize other components that require it (e.g., retrievers might need connection)
            if hasattr(self.retriever, "initialize"):
                logger.info("Initializing Retriever...")
                await self.retriever.initialize()

            # Reranker initialization might be synchronous or done in its __init__
            if (
                self.reranker
                and hasattr(self.reranker, "_initialize_model")
                and not getattr(self.reranker, "_initialized", True)
            ):
                logger.info("Initializing Reranker...")
                # Assuming a synchronous init method for models
                self.reranker._initialize_model()

            # Cache, Embedder, Analyzer usually don't need separate async init
            # unless they load large resources asynchronously.

            if load_existing:
                logger.info("Loading existing notes...")
                await self._load_and_index_notes()

            self._initialized = True
            logger.info("✅ Agentic Memory System initialization complete.")
        except Exception as e:
            logger.critical(f"Failed to initialize memory system: {e}", exc_info=True)
            self._initialized = False  # Ensure state reflects failure
            raise  # Re-raise the exception to signal failure

    async def _load_and_index_notes(self) -> None:
        """Load all notes from the store, populate cache, and rebuild index if retriever requires it."""
        logger.info("Loading notes from store...")
        try:
            all_notes = await self.store.get_all()
            logger.info(f"Loaded {len(all_notes)} notes from store.")

            # Populate cache
            await self.cache.clear()  # Start fresh
            for note in all_notes:
                await self.cache.put(note)
            logger.info(f"Populated cache with {len(all_notes)} notes.")

            # Rebuild retriever index(es) - Check if retriever manages its own index
            if isinstance(self.retriever, IManagesOwnIndex):
                logger.info("Rebuilding index for retriever that manages its own index...")
                try:
                    await self.retriever.rebuild_index(all_notes)
                    logger.info("Retriever index rebuild complete.")
                except Exception as rebuild_err:
                    logger.error(f"Retriever index rebuild failed: {rebuild_err}", exc_info=True)
            else:
                logger.info("Skipping explicit index rebuild (retriever doesn't manage its own index).")

        except Exception as e:
            logger.error(f"Error loading/indexing existing notes: {e}", exc_info=True)
            # Allow initialization to continue but log the error

    # --- CRUD Operations ---

    async def create(self, content: str, **extra_metadata) -> MemoryNote:
        """Create a new memory note, analyze, embed, store, cache, and optionally update retriever index."""
        if not self._initialized:
            raise RuntimeError("Memory system is not initialized. Call initialize() first.")

        note_name = extra_metadata.get("name", "N/A")
        logger.info(f"Creating new memory note (name: {note_name})...")

        # 1. Analyze content (if enabled)
        analysis = {}
        metadata_from_analysis = {}
        if self.config.enable_llm_analysis:
            try:
                logger.debug("Analyzing content...")
                analysis = await self.analyzer.analyze_content(content)
                logger.debug(f"Content analysis result: {analysis}")
                metadata_from_analysis = {
                    "keywords": analysis.get("keywords", []),
                    "summary": analysis.get("summary", ""),
                    "sentiment": analysis.get("sentiment", "neutral"),
                    "importance": analysis.get("importance", 0.5),
                    "type": analysis.get("primary_type", "general"),  # Use 'type' as standard key
                }
            except Exception as e:
                logger.error(f"Content analysis failed: {e}", exc_info=True)

        # TODO: Implement segmentation logic here if self.config.enable_auto_segmentation
        if self.config.enable_auto_segmentation and analysis.get("has_mixed_content", False):
            logger.warning("Auto-segmentation is enabled but not fully implemented yet.")
            # segments = await self.analyzer.segment_content(content)
            # if len(segments) > 1: # Handle parent/child notes...

        # 2. Combine metadata and create MemoryNote
        final_metadata = {**metadata_from_analysis, **extra_metadata}
        note = MemoryNote(content=content, **final_metadata)
        logger.debug(f"Created initial MemoryNote object: {note.id}")

        # 3. Generate embedding
        embedding = None
        storage_task_type = analysis.get("recommended_task_types", {}).get("storage")
        try:
            logger.debug(f"Generating embedding (task type: {storage_task_type})...")
            embedding = await self.embedder.get_embeddings(note.content, task_type=storage_task_type)
            logger.debug("Generated embedding.")
        except Exception as e:
            logger.error(f"Embedding generation failed for {note.id}: {e}", exc_info=True)

        # 4. Store persistently
        try:
            logger.debug(f"Storing note {note.id}...")
            await self.store.add(note, embedding)
            logger.info(f"Stored note {note.id} successfully.")
        except Exception as e:
            logger.error(f"Failed to store note {note.id}: {e}", exc_info=True)
            raise  # Storing is critical

        # 5. Update cache
        try:
            await self.cache.put(note)
            logger.debug(f"Cached note {note.id}.")
        except Exception as e:
            logger.error(f"Failed to cache note {note.id}: {e}", exc_info=True)

        # 6. Update retriever index(es) - Check if retriever manages its own index
        if isinstance(self.retriever, IManagesOwnIndex):
            try:
                await self.retriever.add_document(note.id, note.content, note.metadata)
                logger.debug(f"Added note {note.id} to retriever's managed index.")
            except Exception as e:
                logger.error(f"Failed to add note {note.id} to retriever's managed index: {e}", exc_info=True)
        else:
            logger.debug(f"Skipping explicit add_document for retriever {type(self.retriever).__name__}.")

        logger.info(f"✅ Memory note {note.id} created and processed.")
        return note

    async def get(self, note_id: str) -> Optional[MemoryNote]:
        """Get a memory note by ID, checking cache first, then store."""
        if not self._initialized:
            raise RuntimeError("Memory system is not initialized.")

        logger.debug(f"Getting note {note_id}...")
        note = await self.cache.get(note_id)
        if note:
            logger.info(f"Retrieved note {note_id} from cache.")
            return note

        logger.info(f"Note {note_id} not in cache, checking store...")
        note = await self.store.get_by_id(note_id)
        if note:
            logger.info(f"Retrieved note {note_id} from store.")
            await self.cache.put(note)  # Add to cache
            return note
        else:
            logger.warning(f"Note {note_id} not found in store.")
            return None

    async def update(self, note_id: str, content: Optional[str] = None, **metadata) -> Optional[MemoryNote]:
        """Update a memory note, potentially re-indexing if retriever requires it."""
        if not self._initialized:
            raise RuntimeError("Memory system is not initialized.")

        logger.info(f"Updating memory note {note_id}...")
        note = await self.get(note_id)  # Use self.get to ensure fetch
        if not note:
            logger.error(f"Cannot update: Note {note_id} not found.")
            return None

        content_updated = False
        if content is not None and note.content != content:
            note.content = content
            content_updated = True
            logger.debug(f"Note {note_id} content updated.")

        if metadata:
            note.update_metadata(metadata)  # Handles updated_at
            logger.debug(f"Note {note_id} metadata updated.")
        elif content_updated:
            note.update_metadata({})  # Update timestamp

        embedding = None
        analysis = {}
        if content_updated:
            logger.debug(f"Content changed for {note_id}, re-analyzing and re-embedding...")
            if self.config.enable_llm_analysis:
                try:
                    analysis = await self.analyzer.analyze_content(note.content)
                    logger.debug(f"Re-analysis complete for {note_id}.")
                    # Optional: Auto-update metadata based on re-analysis
                except Exception as e:
                    logger.error(f"Re-analysis failed for {note_id}: {e}")

            storage_task_type = analysis.get("recommended_task_types", {}).get("storage")
            try:
                embedding = await self.embedder.get_embeddings(note.content, task_type=storage_task_type)
                logger.debug(f"Generated updated embedding for {note.id}.")
            except Exception as e:
                logger.error(f"Embedding update failed for {note_id}: {e}")

        # Update store
        try:
            await self.store.update(note, embedding)
            logger.info(f"Stored updated note {note.id}.")
        except Exception as e:
            logger.error(f"Failed to store updated note {note.id}: {e}", exc_info=True)
            raise

        # Update cache
        await self.cache.put(note)

        # Update retriever index if it manages its own
        if isinstance(self.retriever, IManagesOwnIndex):
            try:
                await self.retriever.add_document(note.id, note.content, note.metadata)
                logger.debug(f"Updated note {note.id} in retriever's managed index.")
            except Exception as e:
                logger.error(f"Failed to update note {note.id} in retriever's managed index: {e}")
        else:
            logger.debug(f"Skipping explicit update in retriever {type(self.retriever).__name__}.")

        logger.info(f"✅ Memory note {note_id} updated successfully.")
        return note

    async def delete(self, note_id: str) -> bool:
        """Delete a memory note from store, cache, and retriever index if it manages its own."""
        if not self._initialized:
            raise RuntimeError("Memory system is not initialized.")

        logger.info(f"Deleting memory note {note_id}...")
        deleted_from_store = False
        try:
            deleted_from_store = await self.store.delete(note_id)
            if deleted_from_store:
                logger.info(f"Deleted note {note_id} from store.")
            else:
                logger.warning(f"Note {note_id} not found in store for deletion.")
        except Exception as e:
            logger.error(f"Failed to delete note {note_id} from store: {e}", exc_info=True)

        # Delete from cache regardless
        try:
            await self.cache.delete(note_id)
            logger.debug(f"Deleted note {note_id} from cache.")
        except Exception as e:
            logger.error(f"Failed to delete note {note_id} from cache: {e}")

        # Delete from retriever index(es) - Check if retriever manages its own index
        if isinstance(self.retriever, IManagesOwnIndex):
            try:
                logger.debug(f"Requesting removal of {note_id} from retriever's managed index...")
                await self.retriever.remove_document(note_id)
                logger.debug(f"Processed removal for note {note_id} from retriever's managed index.")
            except Exception as e:
                logger.error(f"Failed to process removal for note {note_id} from retriever's managed index: {e}")
        else:
            logger.debug(f"Skipping explicit remove_document for retriever {type(self.retriever).__name__}.")

        logger.info(f"✅ Memory note {note_id} deletion process completed.")
        return deleted_from_store

    # --- Search Operation ---

    async def search(
        self,
        query: str,
        k: int = 5,
        use_reranker: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None,
        query_task_type_override: Optional[str] = None,
    ) -> List[MemoryNote]:
        """Search for memory notes using retriever, optionally rerank, and fetch full notes."""
        if not self._initialized:
            raise RuntimeError("Memory system is not initialized.")

        logger.info(f"Searching memories (k={k}, rerank={use_reranker}): '{query[:50]}...'")

        # 1. Determine query embedding and task type
        query_embedding = None
        final_query_task_type = query_task_type_override
        effective_metadata_filter = metadata_filter.copy() if metadata_filter else {}

        if not final_query_task_type and self.config.enable_llm_analysis:
            try:
                analysis = await self.analyzer.analyze_content(query)
                final_query_task_type = analysis.get("recommended_task_types", {}).get("query", "RETRIEVAL_QUERY")
                logger.debug(f"Analyzed query task type: {final_query_task_type}")
                # Add implicit filter based on analysis if needed
            except Exception as e:
                logger.error(f"Query analysis failed: {e}")
                final_query_task_type = "RETRIEVAL_QUERY"
        elif not final_query_task_type:
            final_query_task_type = "RETRIEVAL_QUERY"

        if effective_metadata_filter:
            logger.debug(f"Applying metadata filter: {effective_metadata_filter}")

        try:
            query_embedding = await self.embedder.get_embeddings(query, task_type=final_query_task_type)
            logger.debug("Generated query embedding.")
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")

        # 2. Retrieve initial candidates
        initial_k = k * 5 if use_reranker and self.reranker else k
        try:
            candidates: List[SearchResult] = await self.retriever.search(
                query=query, top_k=initial_k, embedding=query_embedding, metadata_filter=effective_metadata_filter
            )
            logger.info(f"Retriever found {len(candidates)} initial candidates.")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return []

        if not candidates:
            logger.warning("No candidates found by retriever.")
            return []

        # 3. Rerank candidates
        reranked_candidates = candidates
        if use_reranker and self.reranker:
            candidates_with_content = [c for c in candidates if c.content is not None]
            if len(candidates_with_content) < len(candidates):
                logger.warning("Some candidates missing content for reranking.")
            if candidates_with_content:
                try:
                    logger.info(f"Reranking {len(candidates_with_content)} candidates...")
                    reranked_subset = await self.reranker.rerank(query, candidates_with_content)
                    reranked_candidates = reranked_subset
                    logger.info(f"Reranking complete. {len(reranked_candidates)} results after rerank.")
                except Exception as e:
                    logger.error(f"Reranking failed: {e}. Using original order for candidates with content.")
                    reranked_candidates = candidates_with_content  # Fallback
            else:
                logger.warning("No candidates had content for reranking.")
        elif use_reranker:
            logger.warning("Reranking requested but no reranker is configured.")

        # 4. Get final top-k IDs and fetch full notes
        final_ids = [res.note_id for res in reranked_candidates[:k]]
        if not final_ids:
            logger.info("Search/reranking yielded no final candidates.")
            return []

        logger.debug(f"Fetching final top {len(final_ids)} notes: {final_ids}")
        fetch_tasks = [self.get(note_id) for note_id in final_ids]
        results_with_none = await asyncio.gather(*fetch_tasks)

        # Filter out None results and maintain order
        final_notes_map = {note.id: note for note in results_with_none if note is not None}
        final_ordered_notes = [final_notes_map[note_id] for note_id in final_ids if note_id in final_notes_map]

        logger.info(f"✅ Search completed. Returning {len(final_ordered_notes)} final notes.")
        return final_ordered_notes

    # --- Utility Methods (Keep get_all*, get_stats, shutdown) ---
    # ... (get_all_notes_from_cache, get_all_notes_from_store, get_stats, shutdown are mostly correct) ...

    async def get_all_notes_from_cache(self) -> List[MemoryNote]:
        """Retrieve all memory notes currently held in the cache."""
        if not self._initialized:
            raise RuntimeError("Memory system is not initialized.")
        return await self.cache.get_all_notes()

    async def get_all_notes_from_store(self) -> List[MemoryNote]:
        """Retrieve all memory notes directly from the persistent store."""
        if not self._initialized:
            raise RuntimeError("Memory system is not initialized.")
        return await self.store.get_all()

    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system (e.g., cache size, store size)."""
        if not self._initialized:
            raise RuntimeError("Memory system is not initialized.")

        stats = {}
        try:
            # Cache size
            if hasattr(self.cache, "get_all_notes"):
                cached_notes = await self.cache.get_all_notes()
                stats["cached_notes"] = len(cached_notes)
            elif hasattr(self.cache, "__len__"):
                stats["cached_notes"] = len(self.cache)

            # Store size
            if hasattr(self.store, "count"):
                stats["stored_notes"] = await self.store.count()
            elif hasattr(self.store, "get_all"):
                all_stored = await self.store.get_all()
                stats["stored_notes"] = len(all_stored)

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            stats["error"] = str(e)

        return stats

    async def shutdown(self, timeout: float = 10.0) -> None:
        """Shutdown components gracefully with a timeout."""
        logger.info(f"Shutting down Agentic Memory System (timeout: {timeout}s)...")
        self._initialized = False  # Set early to prevent new operations

        shutdown_tasks = []
        task_names = []  # Keep track of names for logging

        # Attempt shutdown for each component if it exists
        # We assume components either have shutdown() or close_clients() if relevant

        if self.store:
            logger.info("Initiating Memory Store shutdown...")
            task_names.append("Memory Store")
            shutdown_tasks.append(self._safe_shutdown(self.store, task_names[-1], timeout))

        if self.retriever:
            logger.info("Initiating Retriever shutdown...")
            task_names.append("Retriever")
            shutdown_tasks.append(self._safe_shutdown(self.retriever, task_names[-1], timeout))

        if self.embedder:
            # Prefer shutdown, fallback to close_clients if needed (checked within _safe_shutdown)
            logger.info("Initiating Embedder shutdown/close...")
            task_names.append("Embedder")
            shutdown_tasks.append(self._safe_shutdown(self.embedder, task_names[-1], timeout))

        if self.analyzer and self.analyzer is not self.embedder:  # Avoid double shutdown
            logger.info("Initiating Analyzer shutdown/close...")
            task_names.append("Analyzer")
            shutdown_tasks.append(self._safe_shutdown(self.analyzer, task_names[-1], timeout))

        if self.reranker:
            logger.info("Initiating Reranker shutdown...")
            task_names.append("Reranker")
            shutdown_tasks.append(self._safe_shutdown(self.reranker, task_names[-1], timeout))

        # --- Run all shutdown tasks concurrently ---
        if shutdown_tasks:
            logger.info(f"Waiting for {len(shutdown_tasks)} component(s) to shut down...")
            results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            for result, name in zip(results, task_names):
                if isinstance(result, Exception):
                    # Log error, but don't prevent overall shutdown completion message
                    pass  # Error already logged in _safe_shutdown
                else:
                    logger.info(f"{name} shutdown completed.")
        else:
            logger.info("No components required explicit shutdown.")

        logger.info("✅ Agentic Memory System shutdown process finished.")

    async def _safe_shutdown(self, component: Any, name: str, timeout: float) -> None:
        """Safely shut down a component by calling its shutdown() method."""
        try:
            # Directly attempt to get and call the shutdown method
            shutdown_method = component.shutdown
            if not callable(shutdown_method):
                # Only log if it exists but isn't callable, otherwise AttributeError below handles it
                logger.warning(f"Component '{name}' has a 'shutdown' attribute, but it's not callable.")
                return

            logger.debug(f"Awaiting {name}.shutdown()... (Timeout: {timeout}s)")
            await asyncio.wait_for(shutdown_method(), timeout=timeout)
            logger.info(f"{name} shutdown successful.")
        except AttributeError:
            # This is expected if the component doesn't need/have a shutdown method
            logger.debug(f"Component '{name}' has no shutdown method, assuming no cleanup needed.")
        except asyncio.TimeoutError:
            logger.error(f"Timeout occurred while calling shutdown on {name} after {timeout} seconds.")
        except Exception as e:
            logger.error(f"Error during shutdown for {name}: {e}", exc_info=True)
            # Logged the error, allow gather to continue

    # Removed internal helper methods/classes previously here.
