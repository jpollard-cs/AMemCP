#!/usr/bin/env python3
"""
ChromaDB Memory Store Implementation using AsyncHttpClient.
"""

from typing import Any, Dict, List, Optional

import chromadb
from chromadb import AsyncHttpClient, Collection, GetResult

from amem.core.interfaces import IMemoryStore
from amem.core.mappers import IMemoryNoteMapper
from amem.core.models.memory_note import MemoryNote
from amem.utils.utils import get_chroma_settings, setup_logger

logger = setup_logger(__name__)


# Default HNSW configuration (can be overridden in constructor)
HNSW_CONFIG_DEFAULTS = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 100,
    "hnsw:M": 16,
    "hnsw:search_ef": 20,  # Can be changed after creation
    "hnsw:batch_size": 100,  # Can be changed after creation
    "hnsw:sync_threshold": 1000,  # Can be changed after creation
}


class ChromaStore(IMemoryStore):
    """Storage implementation using ChromaDB AsyncHttpClient.

    Connects to a running ChromaDB server instance.
    Uses a mapper to decouple the MemoryNote domain object from ChromaDB specifics.
    """

    def __init__(
        self,
        collection_name: str,
        host: str,
        port: int,
        mapper: IMemoryNoteMapper,  # Inject the mapper
        hnsw_config: Optional[Dict[str, Any]] = None,
    ):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.mapper = mapper  # Store the injected mapper
        self.settings = get_chroma_settings()
        self.hnsw_config = {**HNSW_CONFIG_DEFAULTS, **(hnsw_config or {})}
        self.client: Optional[AsyncHttpClient] = None
        self.collection: Optional[Collection] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the ChromaDB AsyncHttpClient and collection."""
        if self._initialized:
            logger.debug(f"ChromaStore for '{self.collection_name}' already initialized.")
            return

        logger.info(
            f"Initializing ChromaStore (AsyncHttpClient): Collection='{self.collection_name}', Target={self.host}:{self.port}"
        )
        try:
            # Instantiate AsyncHttpClient directly
            self.client = await chromadb.AsyncHttpClient(host=self.host, port=self.port, settings=self.settings)
            logger.info("ChromaDB AsyncHttpClient created successfully.")
            logger.debug("Checking ChromaDB server heartbeat...")
            await self.client.heartbeat()  # Verify connection
            logger.info("ChromaDB server connection verified.")

            logger.info(f"Getting or creating collection: {self.collection_name}")
            # get_or_create_collection is async for AsyncHttpClient
            self.collection = await self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,  # Embeddings handled externally
                metadata=self.hnsw_config,
            )

            logger.info(f"Obtained ChromaDB collection '{self.collection_name}'.")
            self._initialized = True
            return
        except Exception as e:
            logger.critical(f"Failed to initialize ChromaStore AsyncHttpClient: {e}", exc_info=True)
            self.client = None
            self.collection = None
            self._initialized = False
            raise ConnectionError(f"Could not connect to or initialize ChromaDB at {self.host}:{self.port}") from e

    async def shutdown(self) -> None:
        """Clean up resources (AsyncHttpClient doesn't typically require explicit shutdown)."""
        logger.info(f"Shutting down ChromaStore for collection '{self.collection_name}' (AsyncHttpClient - No-op).")
        # AsyncHttpClient handles connections internally; no explicit close needed usually.
        self.client = None
        self.collection = None
        self._initialized = False

    async def add(self, note: MemoryNote, embedding: Optional[List[float]] = None) -> None:
        """Add a new memory note to the store using the async client."""
        if not self._initialized or not self.collection:
            raise ConnectionError("ChromaStore is not initialized.")

        logger.debug(f"Adding note {note.id} via mapper to Chroma collection '{self.collection_name}'")
        persistence_data = self.mapper.to_persistence(note)

        try:
            # Use async add method directly
            await self.collection.add(
                ids=[persistence_data["id"]],
                embeddings=[embedding] if embedding else None,
                documents=[persistence_data["document"]],
                metadatas=[persistence_data["metadata"]],
            )
            logger.info(f"Successfully added note {note.id} to '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error adding note {note.id} to ChromaDB: {e}", exc_info=True)
            raise

    async def update(self, note: MemoryNote, embedding: Optional[List[float]] = None) -> None:
        """Update an existing memory note (uses upsert) using the async client."""
        if not self._initialized or not self.collection:
            raise ConnectionError("ChromaStore is not initialized.")

        logger.debug(f"Upserting note {note.id} via mapper in Chroma collection '{self.collection_name}'")
        persistence_data = self.mapper.to_persistence(note)

        try:
            # Use async upsert method directly
            await self.collection.upsert(
                ids=[persistence_data["id"]],
                embeddings=[embedding] if embedding else None,
                documents=[persistence_data["document"]],
                metadatas=[persistence_data["metadata"]],
            )
            logger.info(f"Successfully upserted note {note.id} in '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error upserting note {note.id} in ChromaDB: {e}", exc_info=True)
            raise

    async def delete(self, note_id: str) -> bool:
        """Delete a memory note from the store by ID using the async client."""
        if not self._initialized or not self.collection:
            raise ConnectionError("ChromaStore is not initialized.")

        logger.debug(f"Deleting note {note_id} from Chroma collection '{self.collection_name}'")
        try:
            # Use async delete method directly
            await self.collection.delete(ids=[note_id])
            logger.info(f"Successfully deleted note {note_id} from '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting note {note_id} from ChromaDB: {e}", exc_info=True)
            return False

    async def get_by_id(self, note_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note using the async client."""
        if not self._initialized or not self.collection:
            raise ConnectionError("ChromaStore is not initialized.")

        logger.debug(f"Getting note {note_id} from Chroma collection '{self.collection_name}'")
        try:
            # Use async get method directly
            result: Optional[GetResult] = await self.collection.get(ids=[note_id], include=["metadatas", "documents"])

            if not result or not result.get("ids") or not result["ids"]:
                logger.warning(f"Note ID {note_id} not found in '{self.collection_name}'")
                return None

            # Extract raw data
            idx = result["ids"].index(note_id)
            doc_id = result["ids"][idx]
            document = result["documents"][idx] if result.get("documents") else ""
            raw_metadata = result["metadatas"][idx] if result.get("metadatas") else {}

            # Use mapper
            note = self.mapper.to_domain(doc_id, document, raw_metadata)

            logger.info(f"Successfully retrieved and mapped note {note_id} from '{self.collection_name}'")
            return note

        except Exception as e:
            logger.error(f"Error retrieving/mapping note {note_id} from ChromaDB: {e}", exc_info=True)
            return None

    async def get_all(self) -> List[MemoryNote]:
        """Retrieve all notes using the async client."""
        if not self._initialized or not self.collection:
            raise ConnectionError("ChromaStore is not initialized.")

        logger.debug(f"Getting all notes from Chroma collection '{self.collection_name}'")
        notes: List[MemoryNote] = []
        try:
            # Use async get method directly
            all_results: Optional[GetResult] = await self.collection.get(include=["metadatas", "documents"])

            if all_results and all_results.get("ids"):
                count = len(all_results["ids"])
                logger.info(f"Retrieved {count} raw notes from '{self.collection_name}'. Mapping...")
                ids = all_results.get("ids", [])
                documents = all_results.get("documents", [])
                metadatas = all_results.get("metadatas", [])

                for i, note_id in enumerate(ids):
                    content = documents[i] if i < len(documents) else ""
                    raw_metadata = metadatas[i] if i < len(metadatas) else {}

                    # Use mapper
                    try:
                        note = self.mapper.to_domain(note_id, content, raw_metadata)
                        notes.append(note)
                    except Exception as map_err:
                        logger.error(f"Failed to map note {note_id}: {map_err}", exc_info=False)

                logger.info(f"Successfully mapped {len(notes)} notes.")
            else:
                logger.info(f"No notes found in collection '{self.collection_name}'.")

            return notes
        except Exception as e:
            logger.error(f"Error retrieving all notes from ChromaDB: {e}", exc_info=True)
            return []
