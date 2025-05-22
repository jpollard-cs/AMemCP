#!/usr/bin/env python3
"""
Example script demonstrating basic usage of AMemCP with Google Gemini models.

This example shows how to:
- Initialize the memory system with Gemini
- Create, retrieve, update, and delete memories
- Search for memories using semantic search
- Work with metadata and content analysis
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Project imports (after sys.path modification)
from amem.core.cache import InMemoryCache  # noqa: E402
from amem.core.llm_content_analyzer import LLMContentAnalyzer  # noqa: E402
from amem.core.llm_controller import LLMController  # noqa: E402
from amem.core.mappers import MemoryNoteMapper  # noqa: E402
from amem.core.memory_system import AgenticMemorySystem, AgenticMemorySystemConfig  # noqa: E402
from amem.core.rerankers import JinaReranker  # noqa: E402
from amem.core.retrievers import BM25Retriever, ChromaVectorRetriever, EnsembleRetriever  # noqa: E402
from amem.core.stores import ChromaStore  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    """Main function demonstrating AMemCP with Gemini models"""

    # Check for Google API key
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("No Google API key found. Please set the GOOGLE_API_KEY environment variable.")
        logger.info("You can create an API key at: https://console.cloud.google.com/apis/credentials")
        return

    logger.info("🚀 Initializing AMemCP with Google Gemini models...")

    try:
        # Initialize components
        mapper = MemoryNoteMapper()

        # Create store with a unique collection name for this example
        store = ChromaStore(
            collection_name="amem_gemini_example",
            host="localhost",
            port=8001,  # Default ChromaDB port
            mapper=mapper,
        )
        await store.initialize()

        # Create retrievers
        bm25 = BM25Retriever()
        vector = ChromaVectorRetriever(collection=store.collection)
        ensemble = EnsembleRetriever([vector, bm25], [0.7, 0.3])  # Favor vector search

        # Create LLM controller with Gemini
        llm_controller = LLMController(
            backend="gemini",
            model="gemini-2.5-flash-preview-05-20",
            embed_model="gemini-embedding-exp-03-07",
            api_key=google_api_key,
        )

        # Create content analyzer
        analyzer = LLMContentAnalyzer(llm_controller)

        # Optional: Create reranker (may fail if model not available)
        reranker = None
        try:
            reranker = JinaReranker(model_name="jinaai/jina-reranker-v2-base-multilingual")
            if reranker._initialized:
                logger.info("✅ Reranker initialized successfully")
            else:
                logger.warning("⚠️ Reranker failed to initialize, proceeding without it")
                reranker = None
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize reranker: {e}")

        # Create memory system
        memory_system = AgenticMemorySystem(
            memory_store=store,
            cache=InMemoryCache(),
            embedder=llm_controller,
            analyzer=analyzer,
            retriever=ensemble,
            config=AgenticMemorySystemConfig(
                enable_llm_analysis=True,
                enable_auto_segmentation=False,
            ),
            reranker=reranker,
        )

        # Initialize the memory system
        await memory_system.initialize(load_existing=False)  # Start fresh
        logger.info("✅ Memory system initialized successfully")

        # Create some example memories
        logger.info("\n📝 Creating example memories...")

        memories = []

        # Memory 1: About Gemini models
        note1 = await memory_system.create(
            "Google's Gemini models are a family of multimodal AI models that can understand and generate text, code, images, and audio. They're designed to be highly capable and efficient.",
            name="Gemini Overview",
        )
        memories.append(note1)
        logger.info(f"Created memory: {note1.id} - {note1.metadata.get('name', 'Unnamed')}")

        # Memory 2: About embeddings
        note2 = await memory_system.create(
            "Gemini embedding models provide high-quality vector representations of text that capture semantic meaning. They support task-specific optimization for different use cases.",
            name="Gemini Embeddings",
        )
        memories.append(note2)
        logger.info(f"Created memory: {note2.id} - {note2.metadata.get('name', 'Unnamed')}")

        # Memory 3: Code example
        note3 = await memory_system.create(
            """
# Example of using Gemini API
import google.generativeai as genai

genai.configure(api_key="your-api-key")
model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
response = model.generate_content("Explain quantum computing")
print(response.text)
            """.strip(),
            name="Gemini Code Example",
        )
        memories.append(note3)
        logger.info(f"Created memory: {note3.id} - {note3.metadata.get('name', 'Unnamed')}")

        # Display created memories with their analysis
        logger.info("\n📋 Created memories with analysis:")
        for i, memory in enumerate(memories, 1):
            logger.info(f"\nMemory {i}: {memory.metadata.get('name', 'Unnamed')}")
            logger.info(f"  ID: {memory.id}")
            logger.info(f"  Type: {memory.metadata.get('type', 'unknown')}")
            logger.info(f"  Keywords: {memory.metadata.get('keywords', [])}")
            logger.info(f"  Importance: {memory.metadata.get('importance', 'N/A')}")
            logger.info(f"  Content: {memory.content[:100]}{'...' if len(memory.content) > 100 else ''}")

        # Search for memories
        logger.info("\n🔍 Searching for memories...")

        search_queries = [
            "multimodal AI models",
            "vector representations",
            "Python code example",
        ]

        for query in search_queries:
            logger.info(f"\nSearching for: '{query}'")
            results = await memory_system.search(query, k=2, use_reranker=bool(reranker))

            if results:
                logger.info(f"Found {len(results)} results:")
                for j, result in enumerate(results, 1):
                    logger.info(f"  {j}. {result.metadata.get('name', 'Unnamed')} - {result.content[:60]}...")
            else:
                logger.info("  No results found")

        # Update a memory
        logger.info("\n✏️ Updating a memory...")
        updated_note = await memory_system.update(
            note1.id,
            content="Google's Gemini models are state-of-the-art multimodal AI models that can process text, code, images, audio, and video. They represent a significant advancement in AI capabilities.",
            name="Gemini Advanced Overview",
        )

        if updated_note:
            logger.info(f"✅ Updated memory: {updated_note.id}")
            logger.info(f"  New content: {updated_note.content[:100]}...")
            logger.info(f"  Updated keywords: {updated_note.metadata.get('keywords', [])}")
        else:
            logger.error("❌ Failed to update memory")

        # Get memory statistics
        logger.info("\n📊 Memory system statistics:")
        stats = await memory_system.get_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        # Clean up (optional)
        logger.info("\n🧹 Cleaning up memories...")
        for memory in memories:
            success = await memory_system.delete(memory.id)
            if success:
                logger.info(f"  Deleted: {memory.id}")
            else:
                logger.warning(f"  Failed to delete: {memory.id}")

        logger.info("\n✅ Example completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error in example: {e}", exc_info=True)
    finally:
        # Shutdown memory system
        if "memory_system" in locals():
            logger.info("🔄 Shutting down memory system...")
            await memory_system.shutdown()
            logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
