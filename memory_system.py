#!/usr/bin/env python3
"""
Agentic Memory System implementation with LLM integration, retrievers, and reranking.
Uses async I/O throughout for optimal performance.
"""

import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from llm_controller import LLMController
from llm_content_analyzer import LLMContentAnalyzer
from sentence_transformers.cross_encoder import CrossEncoder

# Import async ChromaDB client
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
# Define standard Gemini model names - can be overridden by env vars
DEFAULT_GEMINI_LLM_MODEL = "gemini-1.5-pro-latest" # Or "gemini-1.5-flash-latest"
DEFAULT_GEMINI_EMBED_MODEL = "text-embedding-004" # Or "text-embedding-preview-0409"
DEFAULT_JINA_RERANKER_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
DEFAULT_PERSIST_DIR = "./data/chroma_db"
DEFAULT_COLLECTION_PREFIX = "amem"

class MemoryNote:
    """A single memory note in the memory system."""

    def __init__(self, content: str, id: Optional[str] = None, **metadata):
        """Initialize a memory note.

        Args:
            content: The content of the note
            id: Optional ID for the note (generated if not provided)
            **metadata: Additional metadata for the note
        """
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.created_at = self.metadata.get("created_at", datetime.now().isoformat())
        self.updated_at = self.metadata.get("updated_at", datetime.now().isoformat())
        # Ensure essential metadata exists
        self.metadata.setdefault("keywords", [])
        self.metadata.setdefault("summary", "")
        self.metadata.setdefault("context", "General")
        self.metadata.setdefault("type", "general")
        self.metadata.setdefault("related_notes", [])
        self.metadata.setdefault("sentiment", "neutral")
        self.metadata.setdefault("importance", 0.5)
        self.metadata.setdefault("tags", [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert the note to a dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

class AsyncChromaRetriever:
    """Async wrapper for ChromaDB operations."""
    
    def __init__(self, collection_name: str, persist_directory: str, embedding_function: Any):
        """Initialize async ChromaDB client and collection.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist data
            embedding_function: Function to generate embeddings
        """
        # Initialize async ChromaDB client
        self.client = chromadb.AsyncHttpClient(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            )
        )
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.collection = None
        
    async def initialize(self):
        """Initialize the collection asynchronously."""
        try:
            # Get or create collection
            self.collection = await self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Initialized AsyncChromaRetriever collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing AsyncChromaRetriever: {e}")
            raise
    
    async def add_document(self, document: str, doc_id: str, metadata: Dict[str, Any] = None, embedding: List[float] = None):
        """Add a document to ChromaDB.
        
        Args:
            document: Text content to add
            doc_id: Unique document ID
            metadata: Optional metadata dict
            embedding: Optional pre-computed embedding
        """
        try:
            if embedding is not None:
                # Use pre-computed embedding
                await self.collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[document],
                    metadatas=[metadata] if metadata else None
                )
            else:
                # Let ChromaDB compute embedding
                await self.collection.upsert(
                    ids=[doc_id],
                    documents=[document],
                    metadatas=[metadata] if metadata else None
                )
            logger.debug(f"Added document {doc_id} to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding document {doc_id} to ChromaDB: {e}")
            raise
    
    async def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.
        
        Args:
            doc_id: ID of document to delete
        """
        try:
            await self.collection.delete(ids=[doc_id])
            logger.debug(f"Deleted document {doc_id} from ChromaDB")
        except Exception as e:
            logger.error(f"Error deleting document {doc_id} from ChromaDB: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 5, embedding: List[float] = None):
        """Search for documents in ChromaDB.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            embedding: Optional pre-computed query embedding
            
        Returns:
            List of search results with id, content, and metadata
        """
        try:
            if embedding is not None:
                # Use pre-computed embedding
                results = await self.collection.query(
                    query_embeddings=[embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
            else:
                # Let ChromaDB compute embedding
                results = await self.collection.query(
                    query_texts=[query],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
            
            # Format results
            formatted_results = []
            if results["ids"] and results["ids"][0]:  # Check if we have results
                for i, doc_id in enumerate(results["ids"][0]):
                    formatted_results.append({
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else None
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            raise
    
    async def get_all(self):
        """Get all documents from ChromaDB.
        
        Returns:
            Dictionary with ids, documents, and metadatas
        """
        try:
            return await self.collection.get(include=["metadatas", "documents"])
        except Exception as e:
            logger.error(f"Error getting all documents from ChromaDB: {e}")
            raise

class AsyncBM25Retriever:
    """Async-compatible BM25 retriever."""
    
    def __init__(self):
        """Initialize BM25 retriever."""
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = None
            self.corpus = []
            self.doc_ids = []
            self.initialized = False
        except ImportError:
            logger.error("rank_bm25 not installed. Run: pip install rank-bm25")
            raise
    
    async def add_document(self, document: str, doc_id: str):
        """Add a document to the BM25 index.
        
        Args:
            document: Text content to add
            doc_id: Unique document ID
        """
        from rank_bm25 import BM25Okapi
        
        # Tokenize document
        tokenized_doc = document.lower().split()
        self.corpus.append(tokenized_doc)
        self.doc_ids.append(doc_id)
        
        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.corpus)
        self.initialized = True
    
    async def rebuild_index(self, documents: List[str], doc_ids: List[str]):
        """Rebuild the entire BM25 index.
        
        Args:
            documents: List of document texts
            doc_ids: List of document IDs
        """
        from rank_bm25 import BM25Okapi
        
        self.corpus = [doc.lower().split() for doc in documents]
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi(self.corpus)
        self.initialized = True
    
    async def search(self, query: str, top_k: int = 5):
        """Search for documents using BM25.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            
        Returns:
            List of search results with id, score
        """
        if not self.initialized:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Create (id, score) pairs and sort by score
        id_score_pairs = list(zip(self.doc_ids, scores))
        id_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return [{"id": pair[0], "score": pair[1]} for pair in id_score_pairs[:top_k]]

class AsyncEnsembleRetriever:
    """Async ensemble retriever that combines multiple retrievers."""
    
    def __init__(self, retrievers: List[Any], weights: List[float]):
        """Initialize ensemble retriever.
        
        Args:
            retrievers: List of retriever objects
            weights: List of weights for each retriever (should sum to 1)
        """
        self.retrievers = retrievers
        self.weights = weights
        
        if len(retrievers) != len(weights):
            raise ValueError("Number of retrievers must match number of weights")
    
    async def search(self, query: str, top_k: int = 5, task_type: str = None):
        """Search using ensemble of retrievers.
        
        Args:
            query: Search query text
            top_k: Maximum number of results
            task_type: Optional task type for specialized embedding
            
        Returns:
            List of search results with id, content, score
        """
        all_results = {}
        final_results = []
        
        # Collect results from each retriever
        for i, retriever in enumerate(self.retrievers):
            weight = self.weights[i]
            
            try:
                # Different call signature for ChromaRetriever vs BM25Retriever
                if hasattr(retriever, 'embedding_function') and task_type:  # ChromaRetriever
                    results = await retriever.search(query, top_k=top_k*2)  # Get more for better ensemble
                else:  # BM25Retriever
                    results = await retriever.search(query, top_k=top_k*2)
                
                # Update all_results with weighted scores
                for result in results:
                    doc_id = result["id"]
                    # ChromaRetriever returns distance (lower is better), convert to score
                    if "distance" in result:
                        # Convert distance to score (1 - normalized_distance)
                        score = 1.0 - min(result["distance"], 1.0)
                    else:
                        score = result.get("score", 0.5)  # Default or BM25 score
                    
                    weighted_score = score * weight
                    
                    if doc_id in all_results:
                        all_results[doc_id]["score"] += weighted_score
                    else:
                        all_results[doc_id] = {
                            "id": doc_id,
                            "score": weighted_score,
                            "content": result.get("content", ""),
                            "metadata": result.get("metadata", {})
                        }
            except Exception as e:
                logger.error(f"Error in retriever {i}: {e}")
        
        # Sort by final score and return top_k
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        return final_results[:top_k]

class AgenticMemorySystem:
    """Fully async Agentic Memory System implementation."""

    def __init__(self, 
                 project_name: str = "default",
                 llm_backend: str = "gemini", 
                 llm_model: Optional[str] = None, 
                 embed_model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 persist_directory: Optional[str] = None,
                 reranker_model: Optional[str] = None,
                 jina_api_key: Optional[str] = None, # Jina might require API key for some models/usage
                 **kwargs):
        """Initialize the real memory system.

        Args:
            project_name: Name of the project for partitioning data.
            llm_backend: The LLM provider ('gemini', 'openai', 'ollama', 'mock').
            llm_model: The specific LLM model name.
            embed_model: The specific embedding model name.
            api_key: The API key for the chosen LLM provider (e.g., Google API Key).
            persist_directory: Directory to persist ChromaDB data.
            reranker_model: The Jina reranker model name.
            jina_api_key: Optional API key for Jina services.
            **kwargs: Additional configuration options.
        """
        self.project_name = project_name
        self.persist_directory = persist_directory or os.getenv("PERSIST_DIRECTORY", DEFAULT_PERSIST_DIR)
        
        # Ensure persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Use provided or default model names (prefer env vars if model args are None)
        _llm_model = llm_model or os.getenv("LLM_MODEL", DEFAULT_GEMINI_LLM_MODEL)
        _embed_model = embed_model or os.getenv("EMBED_MODEL", DEFAULT_GEMINI_EMBED_MODEL)
        _reranker_model = reranker_model or os.getenv("JINA_RERANKER_MODEL", DEFAULT_JINA_RERANKER_MODEL)
        _jina_api_key = jina_api_key or os.getenv("JINA_API_KEY")

        logger.info(f"Initializing async memory system for project: {project_name}")
        logger.info(f"  LLM Backend: {llm_backend}")
        logger.info(f"  LLM Model: {_llm_model}")
        logger.info(f"  Embedding Model: {_embed_model}")
        logger.info(f"  Reranker Model: {_reranker_model}")
        logger.info(f"  Persistence Directory: {self.persist_directory}")

        # Initialize LLM Controller and Analyzer
        self.llm_controller = LLMController(
            backend=llm_backend,
            model=_llm_model,
            embed_model=_embed_model,
            api_key=api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") # Pass relevant API key
        )
        self.content_analyzer = LLMContentAnalyzer(self.llm_controller)

        # Initialize Retrievers
        # Ensure unique collection name per project within the persist directory
        collection_name = f"{DEFAULT_COLLECTION_PREFIX}_{project_name.replace(' ', '_').lower()}"
        chroma_persist_path = os.path.join(self.persist_directory, project_name)
        os.makedirs(chroma_persist_path, exist_ok=True)
        logger.info(f"  Chroma Collection: {collection_name} in {chroma_persist_path}")

        self.chroma_retriever = AsyncChromaRetriever(
            collection_name=collection_name,
            persist_directory=chroma_persist_path,
            embedding_function=self.llm_controller.provider # Pass the provider for embeddings
        )
        self.bm25_retriever = AsyncBM25Retriever()
        self.ensemble_retriever = AsyncEnsembleRetriever(
            retrievers=[self.chroma_retriever, self.bm25_retriever],
            weights=[0.6, 0.4] # Example weights, tune as needed
        )

        # Initialize Reranker
        try:
            self.reranker = CrossEncoder(
                _reranker_model,
                automodel_args={"torch_dtype": "auto"}, # Use auto for best available dtype
                trust_remote_code=True # Often needed for custom models like Jina's
            )
            logger.info(f"Initialized reranker: {_reranker_model}")
        except Exception as e:
            logger.error(f"Failed to initialize reranker model '{_reranker_model}': {e}. Reranking will be disabled.")
            self.reranker = None

        # In-memory cache for MemoryNote objects (IDs mapped to objects)
        self.memory_cache: Dict[str, MemoryNote] = {}
        
        # Flag to track initialization
        self.initialized = False
    
    async def initialize(self):
        """Initialize the memory system asynchronously."""
        if self.initialized:
            return
        
        # Initialize ChromaRetriever
        await self.chroma_retriever.initialize()
        
        # Load existing documents
        await self._load_existing_documents()
        
        self.initialized = True
        logger.info("Async memory system initialization complete")

    async def _load_existing_documents(self):
        """Load existing documents from Chroma into BM25 and cache asynchronously."""
        try:
            logger.info("Loading existing documents for BM25 and cache...")
            # Get all documents from Chroma
            all_docs = await self.chroma_retriever.get_all()
            
            if all_docs and all_docs['ids']:
                count = 0
                # Prepare documents for batch BM25 rebuild
                documents = []
                doc_ids = []
                
                for doc_id, content, metadata in zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas']):
                    # Add to document lists
                    documents.append(content)
                    doc_ids.append(doc_id)
                    # Add to in-memory cache
                    self.memory_cache[doc_id] = MemoryNote(content=content, id=doc_id, **metadata)
                    count += 1
                
                # Rebuild BM25 index in one operation
                if documents:
                    await self.bm25_retriever.rebuild_index(documents, doc_ids)
                
                logger.info(f"Loaded {count} existing documents into BM25 and cache.")
            else:
                logger.info("No existing documents found to load.")
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")

    async def create(self, content: str, name: Optional[str] = None, **extra_metadata) -> MemoryNote:
        """Create a new memory note, analyze content, and store it asynchronously.

        Args:
            content: Content of the note.
            name: Optional name/title for the note.
            **extra_metadata: Any additional metadata to store alongside.

        Returns:
            The created MemoryNote object.
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
            
        note_id = str(uuid.uuid4())
        logger.info(f"Creating memory note ID: {note_id}")

        # Analyze content type and determine optimal task type
        # TODO: Make content_analyzer.analyze_content_type async
        analysis = self.content_analyzer.analyze_content_type(content)
        storage_task_type = analysis.get("recommended_task_types", {}).get("storage", "RETRIEVAL_DOCUMENT")
        primary_type = analysis.get("primary_type", "general")
        logger.info(f"  Content analysis: type={primary_type}, storage_task_type={storage_task_type}")

        # Generate initial metadata using LLM
        base_metadata = {
            "keywords": extra_metadata.pop("keywords", []) or ["generated_keyword"],
            "summary": extra_metadata.pop("summary", "") or "Generated summary...",
            "context": extra_metadata.pop("context", "General"),
            "type": primary_type,
            "related_notes": extra_metadata.pop("related_notes", []),
            "sentiment": extra_metadata.pop("sentiment", "neutral"),
            "importance": extra_metadata.pop("importance", 0.5),
            "tags": extra_metadata.pop("tags", []),
            "name": name or extra_metadata.pop("name", None),
            "storage_task_type": storage_task_type,
            "analysis_confidence": analysis.get("confidence", 0.0),
            "has_mixed_content": analysis.get("has_mixed_content", False)
        }
        # Combine base metadata with any extra user-provided metadata
        metadata = {**base_metadata, **extra_metadata}
        
        # Create MemoryNote object
        note = MemoryNote(content=content, id=note_id, **metadata)
        
        # Add document to retrievers
        try:
            # TODO: Make llm_controller.get_embeddings async
            embedding = self.llm_controller.get_embeddings(content, task_type=storage_task_type)
            await self.chroma_retriever.add_document(
                document=content, 
                embedding=embedding,
                metadata=note.metadata, 
                doc_id=note_id
            )
            logger.info(f"  Added to Chroma with task_type: {storage_task_type}")
        except Exception as e:
            logger.error(f"Failed to add document {note_id} to Chroma: {e}")
        
        # Add to BM25 retriever
        try:
            await self.bm25_retriever.add_document(content, note_id)
            logger.info("  Added to BM25.")
        except Exception as e:
            logger.error(f"Failed to add document {note_id} to BM25: {e}")

        # Add to in-memory cache
        self.memory_cache[note_id] = note
        logger.info(f"Created and stored memory: {note_id}")
        return note

    async def get(self, note_id: str) -> Optional[MemoryNote]:
        """Get a memory note by ID from the cache asynchronously.

        Args:
            note_id: ID of the note to retrieve.

        Returns:
            The MemoryNote object or None if not found.
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
            
        note = self.memory_cache.get(note_id)
        if note:
            logger.info(f"Retrieved memory from cache: {note_id}")
        else:
            logger.warning(f"Memory not found in cache: {note_id}")
        return note

    async def update(self, note_id: str, content: str, **metadata) -> Optional[MemoryNote]:
        """Update a memory note's content and metadata asynchronously.

        Args:
            note_id: ID of the note to update.
            content: New content for the note.
            **metadata: Additional metadata fields to update or add.

        Returns:
            The updated MemoryNote object or None if not found.
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
            
        if note_id not in self.memory_cache:
            logger.error(f"Cannot update memory: {note_id} not found in cache.")
            return None

        logger.info(f"Updating memory: {note_id}")
        original_note = self.memory_cache[note_id]
        original_note.content = content
        original_note.metadata.update(metadata) # Update existing metadata dict
        original_note.updated_at = datetime.now().isoformat()

        # Re-analyze content and task type
        # TODO: Make content_analyzer.analyze_content_type async
        analysis = self.content_analyzer.analyze_content_type(content)
        storage_task_type = analysis.get("recommended_task_types", {}).get("storage", "RETRIEVAL_DOCUMENT")
        primary_type = analysis.get("primary_type", "general")
        original_note.metadata["storage_task_type"] = storage_task_type
        original_note.metadata["type"] = primary_type
        original_note.metadata["analysis_confidence"] = analysis.get("confidence", 0.0)
        original_note.metadata["has_mixed_content"] = analysis.get("has_mixed_content", False)
        logger.info(f"  Re-analyzed: type={primary_type}, task_type={storage_task_type}")

        # Update in ChromaDB
        try:
            # TODO: Make llm_controller.get_embeddings async
            embedding = self.llm_controller.get_embeddings(content, task_type=storage_task_type)
            await self.chroma_retriever.add_document(
                document=content,
                embedding=embedding,
                metadata=original_note.metadata,
                doc_id=note_id
            )
            logger.info(f"  Updated in Chroma with task_type: {storage_task_type}")
        except Exception as e:
            logger.error(f"Failed to update document {note_id} in Chroma: {e}")

        # Update in BM25
        try:
            # Collect all documents for BM25 rebuild
            documents = []
            doc_ids = []
            for nid, note_obj in self.memory_cache.items():
                documents.append(note_obj.content)
                doc_ids.append(nid)
            
            # Rebuild BM25 index
            await self.bm25_retriever.rebuild_index(documents, doc_ids)
            logger.info("  Rebuilt BM25 index after update.")
        except Exception as e:
            logger.error(f"Failed to update BM25 index for {note_id}: {e}")

        logger.info(f"Updated memory {note_id} successfully.")
        return original_note

    async def delete(self, note_id: str) -> bool:
        """Delete a memory note asynchronously.

        Args:
            note_id: ID of the note to delete.

        Returns:
            True if deletion was successful, False otherwise.
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
            
        if note_id not in self.memory_cache:
            logger.warning(f"Memory not found for deletion: {note_id}")
            return False

        logger.info(f"Deleting memory: {note_id}")

        # Delete from ChromaDB
        try:
            await self.chroma_retriever.delete_document(note_id)
            logger.info("  Deleted from Chroma.")
        except Exception as e:
            logger.error(f"Failed to delete document {note_id} from Chroma: {e}")

        # Delete from memory cache
        del self.memory_cache[note_id]
        
        # Rebuild BM25 index
        try:
            # Collect all remaining documents
            documents = []
            doc_ids = []
            for nid, note_obj in self.memory_cache.items():
                documents.append(note_obj.content)
                doc_ids.append(nid)
            
            # Rebuild BM25 index
            await self.bm25_retriever.rebuild_index(documents, doc_ids)
            logger.info("  Rebuilt BM25 index after deletion.")
        except Exception as e:
            logger.error(f"Failed to update BM25 index after deleting {note_id}: {e}")

        logger.info(f"Deleted memory {note_id} from all systems.")
        return True

    async def search(self, query: str, k: int = 5, use_reranker: bool = True) -> List[MemoryNote]:
        """Search for memory notes asynchronously.

        Args:
            query: Search query.
            k: Maximum number of results.
            use_reranker: Whether to use the reranker if available.

        Returns:
            List of relevant MemoryNote objects, sorted by relevance.
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
            
        logger.info(f"Searching memories for: '{query}' (top_k={k}, reranker={'enabled' if use_reranker else 'disabled'})" )
        
        # 1. Analyze query to get optimal task type
        # TODO: Make content_analyzer.get_optimal_task_type async
        query_task_type = self.content_analyzer.get_optimal_task_type(query, is_query=True)
        logger.info(f"  Query analysis: task_type={query_task_type}")
        
        # 2. Get initial candidates from Ensemble Retriever
        initial_k = k * 5 if use_reranker and self.reranker else k
        try:
            candidates = await self.ensemble_retriever.search(
                query, 
                top_k=initial_k, 
                task_type=query_task_type
            )
            logger.info(f"  Initial candidates retrieved: {len(candidates)}")
        except Exception as e:
            logger.error(f"Error during ensemble search: {e}")
            return []
            
        if not candidates:
            return []
            
        # Prepare for reranking
        candidate_ids = [candidate["id"] for candidate in candidates]
        
        # With reranking
        if use_reranker and self.reranker and candidates:
            try:
                logger.info(f"  Reranking {len(candidates)} candidates...")
                
                # Create query-document pairs for reranking
                rerank_pool = [(query, candidate["content"]) for candidate in candidates]
                
                # TODO: Make reranker.predict async
                scores = self.reranker.predict(rerank_pool)
                
                # Combine candidates with scores and sort
                reranked_candidates = sorted(
                    zip(candidate_ids, scores), 
                    key=lambda item: item[1], 
                    reverse=True
                )
                logger.info("  Reranking completed.")
                
                # Get top k reranked IDs
                final_ids = [item[0] for item in reranked_candidates[:k]]
                
            except Exception as e:
                logger.error(f"Error during reranking: {e}. Falling back to initial ranking.")
                final_ids = candidate_ids[:k]
        else:
            # Use initial ranking if reranker is disabled or unavailable
            final_ids = candidate_ids[:k]
            if use_reranker and not self.reranker:
                 logger.warning("  Reranker requested but not available/initialized.")

        # 4. Retrieve full MemoryNote objects from cache for the final IDs
        final_results = []
        for note_id in final_ids:
            note = self.memory_cache.get(note_id)
            if note:
                final_results.append(note)
            else:
                logger.warning(f"Note ID {note_id} from search results not found in cache.")

        logger.info(f"Search completed. Returning {len(final_results)} results.")
        return final_results
    
    async def get_all(self) -> List[MemoryNote]:
        """Get all memory notes asynchronously.
        
        Returns:
            List of all MemoryNote objects.
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
            
        return list(self.memory_cache.values())
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system asynchronously.
        
        Returns:
            Dictionary with statistics like total count and average content length.
        """
        # Ensure system is initialized
        if not self.initialized:
            await self.initialize()
            
        memories = self.memory_cache
        
        total = len(memories)
        if total == 0:
            return {"total_memories": 0, "average_content_length": 0.0}
            
        avg_length = sum(len(m.content) for m in memories.values()) / total
        
        return {
            "total_memories": total,
            "average_content_length": round(avg_length, 1)
        }
