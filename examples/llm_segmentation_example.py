#!/usr/bin/env python3
"""
Example script to demonstrate LLM-based content analysis and segmentation.
Shows how mixed content is intelligently segmented and processed.
"""

import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import memory system and content analyzer
from amem.core import AgenticMemorySystem

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample mixed content that combines code, explanations, and a question
MIXED_CONTENT_SAMPLE = """
# Fibonacci Implementation Guide

This guide demonstrates different ways to implement the Fibonacci sequence in Python.

## Recursive Implementation

The most straightforward approach is a recursive implementation:

```python
def fibonacci_recursive(n):
    '''Return the nth Fibonacci number using recursion.'''
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
```

While elegant, this approach has exponential time complexity O(2^n) due to repeated calculations.

## Improved Implementation with Memoization

We can significantly improve the performance using memoization:

```python
def fibonacci_memoized(n, memo={}):
    '''Return the nth Fibonacci number using memoization.'''
    if n in memo:
        return memo[n]
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        memo[n] = fibonacci_memoized(n-1, memo) + fibonacci_memoized(n-2, memo)
        return memo[n]
```

This brings the time complexity down to O(n).

## Iterative Implementation

For the best performance, we can use an iterative approach:

```python
def fibonacci_iterative(n):
    '''Return the nth Fibonacci number using iteration.'''
    if n <= 0:
        return 0
    elif n == 1:
        return 1

    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b, a + b
    return b
```

The iterative approach has O(n) time complexity and O(1) space complexity.

## Questions to Consider

How would you implement a Fibonacci generator that yields Fibonacci numbers indefinitely?

How would the time and space complexity change when using generators?
"""


async def main():
    """Run the example script"""
    # Check if we have an API key
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Please set either OPENAI_API_KEY or GOOGLE_API_KEY environment variable.")
        return

    # Determine which backend to use based on available API keys
    llm_backend = "openai" if os.environ.get("OPENAI_API_KEY") else "gemini"

    # Initialize memory system with advanced features enabled
    memory_system = AgenticMemorySystem(
        project_name="segmentation_example",
        llm_backend=llm_backend,
        api_key=api_key,
        enable_llm_analysis=True,
        enable_auto_segmentation=True,
    )

    # Initialize the memory system
    await memory_system.initialize()

    logger.info(f"Using {llm_backend} backend for content analysis")

    # First, store some simple content to demonstrate content analysis
    logger.info("\n=== Storing simple contents with auto-analysis ===")

    # Store text content
    logger.info("Creating text memory...")
    text_memory = await memory_system.create(
        "Python is a versatile programming language known for its readability and simplicity.",
        name="Python Description",
    )
    print(f"Created text memory: {text_memory.id}")
    print(f"Detected content type: {text_memory.metadata.get('type', 'unknown')}")

    # Store code content
    logger.info("\nCreating code memory...")
    code_memory = await memory_system.create(
        """
def hello_world():
    print("Hello, world!")

hello_world()
        """,
        name="Hello World Function",
    )
    print(f"Created code memory: {code_memory.id}")
    print(f"Detected content type: {code_memory.metadata.get('type', 'unknown')}")
    print(f"Storage task type: {code_memory.metadata.get('storage_task_type', 'unknown')}")

    # Store question content
    logger.info("\nCreating question memory...")
    question_memory = await memory_system.create(
        "What is the best way to handle exceptions in Python?", name="Python Exception Question"
    )
    print(f"Created question memory: {question_memory.id}")
    print(f"Detected content type: {question_memory.metadata.get('type', 'unknown')}")

    # Now demonstrate auto-segmentation with mixed content
    logger.info("\n=== Storing mixed content with auto-segmentation ===")

    # Store the complex mixed content
    mixed_memory = await memory_system.create(MIXED_CONTENT_SAMPLE, name="Fibonacci Guide")

    # Check if segmentation happened
    if "segment_ids" in mixed_memory.metadata:
        segment_ids = json.loads(mixed_memory.metadata["segment_ids"])
        print(f"\nContent was automatically segmented into {len(segment_ids)} parts")

        # Display information about each segment
        print("\nSegments created:")
        for i, segment_id in enumerate(segment_ids):
            segment = await memory_system.get(segment_id)
            if segment:
                print(f"  Segment {i+1}: {segment.metadata.get('name', 'Unnamed')}")
                print(f"    Type: {segment.metadata.get('type', 'unknown')}")
                print(f"    Content length: {len(segment.content)} characters")
                print(f"    First 50 chars: {segment.content[:50].strip()}...")
    else:
        print("Content was not segmented (this might indicate an issue)")

    # Try searching with different queries to demonstrate content-specific optimizations
    logger.info("\n=== Searching with content-specific optimizations ===")

    queries = [
        "How do I implement a Fibonacci sequence?",
        "recursive fibonacci function python",
        "What is the time complexity of fibonacci implementations?",
        "fibonacci generator in python",
    ]

    for query in queries:
        logger.info(f"Searching for: {query}")

        # Search with LLM analysis enabled (will auto-detect query type and best task type)
        results = await memory_system.search(query=query, k=2, use_reranker=True)

        print(f"\nResults for query: '{query}'")
        for i, memory in enumerate(results):
            print(f"{i+1}. {memory.metadata.get('name', 'Unnamed')}")
            if len(memory.content) > 100:
                print(f"   {memory.content[:100].strip()}...")
            else:
                print(f"   {memory.content.strip()}")

    # Clean up
    logger.info("\n=== Cleaning up ===")
    await memory_system.delete(text_memory.id)
    await memory_system.delete(code_memory.id)
    await memory_system.delete(question_memory.id)
    await memory_system.delete(mixed_memory.id)

    # Clean up segments
    if "segment_ids" in mixed_memory.metadata:
        segment_ids = json.loads(mixed_memory.metadata["segment_ids"])
        for segment_id in segment_ids:
            await memory_system.delete(segment_id)

    logger.info("\nExample completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
