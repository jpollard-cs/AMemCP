#!/usr/bin/env python3
"""
Example script to demonstrate different Gemini embedding task types.
Shows how the same content is embedded differently based on task type.
"""

import logging
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import memory system
from memory_system import AgenticMemorySystem, detect_content_type

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample texts
CODE_SAMPLE = """
def fibonacci(n):
    '''Return the nth Fibonacci number.'''
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def main():
    print(fibonacci(10))

if __name__ == "__main__":
    main()
"""

QUESTION_SAMPLE = "How do I implement a recursive function to calculate Fibonacci numbers in Python?"

DOCUMENT_SAMPLE = """
The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones,
usually starting with 0 and 1. It was named after the Italian mathematician Leonardo of Pisa, known as Fibonacci.
The sequence commonly starts with 0 and 1, although some authors start the sequence with 1 and 1 or sometimes
even with 1 and 2. The first few values in the sequence are: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144.
"""


def main():
    """Run the example script"""
    # Check if we have a Google API key
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error("Please set the GOOGLE_API_KEY environment variable.")
        logger.error("You can get a key from https://makersuite.google.com/app/apikey")
        return

    # Initialize memory system
    memory_system = AgenticMemorySystem(
        project_name="task_types_example",
        llm_backend="gemini",
        llm_model="gemini-2.5-flash-preview-05-20",
        embed_model="gemini-embedding-exp-03-07",
        api_key=os.environ.get("GOOGLE_API_KEY"),
    )

    logger.info("Detecting content types...")

    # Detect content types
    code_type, code_doc_task, code_query_task = detect_content_type(CODE_SAMPLE)
    question_type, question_doc_task, question_query_task = detect_content_type(QUESTION_SAMPLE)
    document_type, document_doc_task, document_query_task = detect_content_type(DOCUMENT_SAMPLE)

    logger.info(f"Code sample detected as {code_type} with task types: {code_doc_task} / {code_query_task}")
    logger.info(
        f"Question sample detected as {question_type} with task types: {question_doc_task} / {question_query_task}"
    )
    logger.info(
        f"Document sample detected as {document_type} with task types: {document_doc_task} / {document_query_task}"
    )

    # Get embeddings with different task types
    embeddings = {}

    # Code sample embeddings
    logger.info("Generating embeddings for code sample with different task types...")
    embeddings["code_default"] = memory_system.llm_controller.get_embeddings(CODE_SAMPLE)
    embeddings["code_document"] = memory_system.llm_controller.get_embeddings(
        CODE_SAMPLE, task_type="RETRIEVAL_DOCUMENT"
    )
    embeddings["code_query"] = memory_system.llm_controller.get_embeddings(
        CODE_SAMPLE, task_type="CODE_RETRIEVAL_QUERY"
    )
    embeddings["code_retrieval"] = memory_system.llm_controller.get_embeddings(
        CODE_SAMPLE, task_type="RETRIEVAL_DOCUMENT"
    )

    # Question sample embeddings
    logger.info("Generating embeddings for question with different task types...")
    embeddings["question_default"] = memory_system.llm_controller.get_embeddings(QUESTION_SAMPLE)
    embeddings["question_retrieval"] = memory_system.llm_controller.get_embeddings(
        QUESTION_SAMPLE, task_type="RETRIEVAL_QUERY"
    )
    embeddings["question_code"] = memory_system.llm_controller.get_embeddings(
        QUESTION_SAMPLE, task_type="CODE_RETRIEVAL_QUERY"
    )

    # Document sample embeddings
    logger.info("Generating embeddings for document with different task types...")
    embeddings["document_default"] = memory_system.llm_controller.get_embeddings(DOCUMENT_SAMPLE)
    embeddings["document_retrieval"] = memory_system.llm_controller.get_embeddings(
        DOCUMENT_SAMPLE, task_type="RETRIEVAL_DOCUMENT"
    )

    # Calculate similarities between different embeddings
    logger.info("Calculating similarities between different embeddings...")

    # Convert to numpy arrays for similarity calculation
    embedding_arrays = {k: np.array(v).reshape(1, -1) for k, v in embeddings.items()}

    # Calculate similarities
    print("\nSimilarity Matrix (cosine similarity):")
    print("----------------------------------------------------------")
    print("                    |  Code-Doc  |  Question  |  Document  ")
    print("----------------------------------------------------------")

    # Format code embeddings similarities
    code_doc_sim = cosine_similarity(embedding_arrays["code_document"], embedding_arrays["question_retrieval"])[0][0]
    code_doc_doc_sim = cosine_similarity(embedding_arrays["code_document"], embedding_arrays["document_retrieval"])[0][
        0
    ]
    print(f"Code (DOCUMENT)    |     -      |  {code_doc_sim:.6f}  |  {code_doc_doc_sim:.6f}")

    # Format question embeddings similarities
    q_code_sim = cosine_similarity(embedding_arrays["question_retrieval"], embedding_arrays["code_document"])[0][0]
    q_doc_sim = cosine_similarity(embedding_arrays["question_retrieval"], embedding_arrays["document_retrieval"])[0][0]
    print(f"Question (QUERY)   |  {q_code_sim:.6f}  |     -      |  {q_doc_sim:.6f}")

    # Format document embeddings similarities
    doc_code_sim = cosine_similarity(embedding_arrays["document_retrieval"], embedding_arrays["code_document"])[0][0]
    doc_q_sim = cosine_similarity(embedding_arrays["document_retrieval"], embedding_arrays["question_retrieval"])[0][0]
    print(f"Document (DOCUMENT)|  {doc_code_sim:.6f}  |  {doc_q_sim:.6f}  |     -     ")
    print("----------------------------------------------------------")

    # Now test with memory system
    logger.info("\nCreating memories using specialized task types...")

    # Create memories with specific task types
    code_memory = memory_system.create(CODE_SAMPLE, name="Fibonacci function")
    question_memory = memory_system.create(QUESTION_SAMPLE, name="Fibonacci question")
    document_memory = memory_system.create(DOCUMENT_SAMPLE, name="Fibonacci explanation")

    # Wait for memories to be indexed
    time.sleep(1)

    # Search with different query types
    logger.info("\nSearching for memories with question as query...")
    results = memory_system.search(QUESTION_SAMPLE, k=3)
    print("\nResults when searching with question:")
    for i, memory in enumerate(results):
        print(f"{i+1}. {memory.metadata.get('name', 'Unnamed')} - {memory.content[:50]}...")

    logger.info("\nSearching for memories with code as query...")
    results = memory_system.search(CODE_SAMPLE, k=3)
    print("\nResults when searching with code snippet:")
    for i, memory in enumerate(results):
        print(f"{i+1}. {memory.metadata.get('name', 'Unnamed')} - {memory.content[:50]}...")

    logger.info("\nExample completed successfully!")


if __name__ == "__main__":
    main()
