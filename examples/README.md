# AMemCP Examples

This directory contains example scripts that demonstrate various features and capabilities of the AMemCP (Agentive Memory via Context Protocol) system.

## Available Examples

### 1. Gemini Example (`gemini_example.py`)

Demonstrates basic usage of the AMemCP system with Google's Gemini models:

- Creating, retrieving, updating, and deleting memories
- Searching for memories using embeddings
- Working with the Gemini API

**Requirements:** Google API key set as `GOOGLE_API_KEY` environment variable

**Run with:**

```bash
python examples/gemini_example.py
```

### 2. Gemini Task Types Example (`gemini_task_types_example.py`)

Demonstrates how different Gemini embedding task types affect embedding quality for different content types:

- Shows how the same content is embedded differently based on task type
- Compares embedding similarity across different task types
- Illustrates optimal task type selection for different content

**Requirements:** Google API key set as `GOOGLE_API_KEY` environment variable

**Run with:**

```bash
python examples/gemini_task_types_example.py
```

### 3. LLM Segmentation Example (`llm_segmentation_example.py`)

Demonstrates LLM-based content analysis and intelligent segmentation:

- Shows how mixed content (code, text, questions) is analyzed and segmented
- Illustrates content type detection and confidence scoring
- Demonstrates how segmented content can be stored as separate memories
- Shows optimal task type selection for different query types

**Requirements:** Either OpenAI API key (`OPENAI_API_KEY`) or Google API key (`GOOGLE_API_KEY`)

**Run with:**

```bash
python examples/llm_segmentation_example.py
```

## Relationship to Main Project

These examples complement the TypeScript client located in the `client/` directory and the Python test client in `tests/test_client.py`. While the examples demonstrate specific features of the AMemCP system, the clients provide more comprehensive interfaces for interacting with the AMemCP server.

## Getting Started

1. Make sure you have the required API keys set in your environment variables or `.env` file
2. Install all dependencies from the project's main requirements
3. Run any example using the commands shown above 