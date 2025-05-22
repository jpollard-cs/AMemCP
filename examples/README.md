# AMemCP Examples

This directory contains usage examples for the AMemCP library.

## Available Examples

### `gemini_example.py`

A comprehensive example demonstrating how to use AMemCP with Google Gemini models:

- **Memory System Setup**: Initialize the memory system with Gemini LLM backend
- **Document Creation**: Create memory notes with automatic content analysis
- **Search Operations**: Perform semantic and keyword-based searches
- **Advanced Features**: Demonstrates reranking, task-type optimization, and async operations

**Requirements:**
- Google API key (set `GOOGLE_API_KEY` environment variable)
- All dependencies from `requirements.txt`

**Usage:**
```bash
export GOOGLE_API_KEY="your-api-key-here"
python examples/gemini_example.py
```

**What it demonstrates:**
- Creating and searching memories
- Different search strategies (vector + BM25 ensemble)
- LLM-powered content analysis
- Reranking for improved relevance
- Async/await patterns throughout

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Set up your API key: `export GOOGLE_API_KEY="your-key"`
3. Run the example: `python examples/gemini_example.py`

The example includes detailed comments explaining each step and concept.
