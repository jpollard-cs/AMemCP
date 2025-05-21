#!/usr/bin/env python3
"""
Example script demonstrating how to use the A-Mem system with Google Gemini models
"""

import logging
import os
import sys
from pathlib import Path

import dotenv

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Import A-Mem modules
from amem.core import AgenticMemorySystem

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function for the example script"""
    # Check for Google API key
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        logger.error("No Google API key found. Please set the GOOGLE_API_KEY environment variable.")
        logger.info("You can create an API key at: https://console.cloud.google.com/apis/credentials")
        return

    # Initialize memory system with Gemini models
    logger.info("Initializing memory system with Google Gemini models...")
    memory_system = AgenticMemorySystem(
        project_name="gemini_example",
        llm_backend="gemini",
        llm_model="gemini-2.5-flash-preview-05-20",
        embed_model="gemini-embedding-exp-03-07",
        api_key=google_api_key,
    )

    # Create memories
    logger.info("Creating memories...")
    notes = []

    # Add some example memories
    note1 = memory_system.create("Google's Gemini models offer a state-of-the-art alternative to OpenAI's GPT models.")
    notes.append(note1)

    note2 = memory_system.create(
        "Gemini models are capable of understanding both text and images, making them suitable for multimodal tasks."
    )
    notes.append(note2)

    note3 = memory_system.create("The Gemini API provides embeddings as well as text generation capabilities.")
    notes.append(note3)

    # Display created memories
    logger.info("Created memories:")
    for i, note in enumerate(notes):
        logger.info(f"Memory {i+1}:")
        logger.info(f"  ID: {note.id}")
        logger.info(f"  Content: {note.content}")
        logger.info(f"  Keywords: {note.metadata.get('keywords', [])}")
        logger.info(f"  Context: {note.metadata.get('context', '')}")

    # Search for memories
    query = "embeddings API"
    logger.info(f"\nSearching for: '{query}'")

    results = memory_system.search(query, top_k=2)

    logger.info(f"Search results for '{query}':")
    for i, result in enumerate(results):
        logger.info(f"Result {i+1}:")
        logger.info(f"  ID: {result.id}")
        logger.info(f"  Content: {result.content}")

    # Update a memory
    logger.info("\nUpdating memory...")
    updated_note = memory_system.update(
        note_id=note1.id,
        content="Google's Gemini models are multimodal AI models developed as an alternative to OpenAI's GPT models.",
    )

    logger.info(f"Updated memory:")
    logger.info(f"  ID: {updated_note.id}")
    logger.info(f"  Content: {updated_note.content}")
    logger.info(f"  Keywords: {updated_note.metadata.get('keywords', [])}")

    # Clean up - optional
    logger.info("\nCleaning up memories...")
    for note in notes:
        memory_system.delete(note.id)

    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main()
