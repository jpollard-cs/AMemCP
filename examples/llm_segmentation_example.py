#!/usr/bin/env python3
"""
Example script to demonstrate LLM-based content analysis and segmentation.
Shows how mixed content is intelligently segmented and processed.
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_content_analyzer import LLMContentAnalyzer

# Import memory system and content analyzer
from memory_system import AgenticMemorySystem

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


def main():
    """Run the example script"""
    # Check if we have an API key
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("Please set either OPENAI_API_KEY or GOOGLE_API_KEY environment variable.")
        return

    # Determine which backend to use based on available API keys
    llm_backend = "openai" if os.environ.get("OPENAI_API_KEY") else "gemini"

    # Initialize memory system
    memory_system = AgenticMemorySystem(project_name="segmentation_example", llm_backend=llm_backend, api_key=api_key)

    # Initialize content analyzer
    content_analyzer = LLMContentAnalyzer(memory_system.llm_controller)

    logger.info(f"Using {llm_backend} backend for content analysis")

    # Analyze the mixed content
    logger.info("Analyzing mixed content...")
    analysis = content_analyzer.analyze_content_type(MIXED_CONTENT_SAMPLE)

    print("\nContent Analysis:")
    print(f"Primary type: {analysis['primary_type']}")
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"Has mixed content: {analysis['has_mixed_content']}")
    print("\nContent type breakdown:")
    for content_type, proportion in analysis["types"].items():
        if proportion > 0:
            print(f"  - {content_type}: {proportion:.2f}")

    print("\nRecommended task types:")
    print(f"  - For storage: {analysis['recommended_task_types']['storage']}")
    print(f"  - For queries: {analysis['recommended_task_types']['query']}")

    # Segment the mixed content
    logger.info("Segmenting mixed content...")
    segments = content_analyzer.segment_content(MIXED_CONTENT_SAMPLE)

    print(f"\nContent was segmented into {len(segments)} parts:")
    for i, segment in enumerate(segments):
        print(f"\nSegment {i+1} - {segment['type']} (task type: {segment['task_type']})")
        if "metadata" in segment and "subtitle" in segment["metadata"] and segment["metadata"]["subtitle"]:
            print(f"Subtitle: {segment['metadata']['subtitle']}")
        print(f"Content length: {len(segment['content'])} characters")
        print(f"First 100 chars: {segment['content'][:100].strip()}")

    # Store segments as separate memories
    logger.info("Storing segments as separate memories...")
    memory_ids = []

    for i, segment in enumerate(segments):
        # Create a name from metadata if available
        name = segment["metadata"].get("subtitle", f"Segment {i+1} - {segment['type']}")

        # Store the segment with its detected task type
        memory = memory_system.create(
            segment["content"], name=name, content_type=segment["type"], task_type=segment["task_type"]
        )
        memory_ids.append(memory.id)
        logger.info(f"Created memory: {name}")

    # Wait for indexing
    time.sleep(1)

    # Try searching with different queries
    logger.info("\nSearching with different queries...")

    queries = [
        "How do I implement a Fibonacci sequence?",
        "recursive fibonacci function python",
        "What is the time complexity of fibonacci implementations?",
        "fibonacci generator in python",
    ]

    for query in queries:
        logger.info(f"Searching for: {query}")
        # Get optimal task type for this query
        task_type = content_analyzer.get_optimal_task_type(query, is_query=True)
        logger.info(f"Using task type: {task_type}")

        # Search with the optimal task type
        results = memory_system.search(query, k=3)

        print(f"\nResults for query: '{query}'")
        for i, memory in enumerate(results):
            print(f"{i+1}. {memory.metadata.get('name', 'Unnamed')} - {memory.content[:50]}...")

    # Clean up
    for memory_id in memory_ids:
        memory_system.delete(memory_id)
    logger.info("\nExample completed successfully!")


if __name__ == "__main__":
    main()
