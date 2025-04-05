# AMemCP - Agentic Memory System with Model Context Protocol built on top of AMem

An agentic memory system that provides persistent storage and semantic search capabilities for AI agents, with connections via the Model Context Protocol (MCP).

This is still a work in progress and much it it was "vibe coded". Its now being formalized / cleaned-up. Contributions are welcome.

## A-Mem: Agentic Memory System

A-Mem is a novel agentic memory system for Language Model (LLM) agents that dynamically organizes memories based on semantic structure, relationships, and agent-specific considerations, optimizing for effective retrieval and informed decision-making.

## Key Features

- **Dynamic Memory Organization**: Memories are stored with rich metadata and organized dynamically rather than in static structures
- **Intelligent Indexing**: Utilizes hybrid retrieval with vector embedding similarity and BM25 lexical search
- **Comprehensive Note Generation**: Automatically extracts keywords, summaries, and context
- **Interconnected Knowledge Network**: Creates links between related memories
- **Continuous Memory Evolution**: Updates memory metadata when new related information is encountered

## Framework

A-Mem employs a flexible framework to organize memories:

1. **Memory Notes**: Atomic units with content and metadata
2. **Metadata Structure**: Includes keywords, context, type, timestamps, related notes, sentiment, and importance
3. **Hybrid Search**: Combines semantic and keyword-based search
4. **Memory Evolution**: Updates metadata as related information is encountered

## Latest Updates

- **ChromaDB 1.0.0 Support**: Updated to use the latest ChromaDB with improved performance and stability
- **Async HTTP Client**: Using the dedicated ChromaDB client package for HTTP connections
- **Colorful Logging**: Enhanced logging with colorful output and emojis for better readability
- **TypeScript Client**: Added a TypeScript client for testing the MCP server

## Project Structure

- `sse_helper.py` - Helper utilities for SSE communication
- `amem/server/mcp_fastmcp_server.py` - Main FastMCP server implementation
- `amem/core/memory_system.py` - Core memory system with async retrieval capabilities
- `amem/utils/utils.py` - Shared utilities for logging and constants
- `amem/mcp_client.py` - Python client for connecting to the MCP server
- `client/` - TypeScript client implementation for connecting to the MCP server
- `examples/`: Example scripts demonstrating various AMemCP features (see [Examples README](examples/README.md))
- `tests/`: Test suite for AMemCP server and client

## Shared Utilities

The `utils.py` module provides common functionality used across multiple modules:

- Colorful logging setup with `colorlog`
- Common constants for model names and default settings
- Shared configuration handling
- ChromaDB settings management

### Usage

```python
# Import utilities
from utils import setup_logger, DEFAULT_PERSIST_DIR, get_chroma_settings

# Set up a logger for your module
logger = setup_logger(__name__)

# Use the logger
logger.info("This is an info message")
logger.debug("This is a debug message")
logger.warning("This is a warning message")
logger.error("This is an error message")

# Get ChromaDB settings
from chromadb import Client
client = Client(get_chroma_settings("/path/to/persist"))
```

## Environment Variables

The application can be configured using the following environment variables:

- `PROJECT_NAME` - Name of the project (default: "default")
- `LLM_BACKEND` - LLM provider to use (default: "gemini")
- `LLM_MODEL` - LLM model name
- `EMBED_MODEL` - Embedding model name
- `GOOGLE_API_KEY` - Google API key for Gemini models
- `OPENAI_API_KEY` - OpenAI API key for OpenAI models
- `PERSIST_DIRECTORY` - Directory to persist data
- `RERANKER_MODEL` - reranker model name

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Creating Memories

```python
from memory_system import AgenticMemorySystem

# Initialize the memory system
memory_system = await AgenticMemorySystem(
    # Optional configuration parameters
    project_name="my_project",
    llm_backend="openai",  # or "gemini", "ollama", "mock"
    llm_model="gpt-4"      # model name specific to the backend
)

# Create a memory note
note = memory_system.create("The Golden Gate Bridge is located in San Francisco.")
print(f"Created note with ID: {note.id}")
```

### Reading Memories

```python
# Retrieve a memory by ID
note = await memory_system.read("note_id_here")
if note:
    print(f"Content: {note.content}")
    print(f"Keywords: {note.metadata.get('keywords', [])}")
    print(f"Created at: {note.created_at}")
```

### Updating Memories

```python
# Update a memory's content
updated_note = await memory_system.update("note_id_here", "Updated content here")
print(f"Updated note: {updated_note.content}")
```

### Deleting Memories

```python
# Delete a memory
success = await memory_system.delete("note_id_here")
print(f"Deletion successful: {success}")
```

### Searching Memories

```python
# Basic semantic search
results = awaitmemory_system.search("San Francisco landmarks", top_k=5)
for note in results:
    print(f"ID: {note.id}, Content: {note.content}")

# Hybrid search (combining semantic and keyword search)
results = await memory_system.search(
    "San Francisco landmarks",
    top_k=5,
    keywords=["bridge", "tourist"]
)
```

## Features

- 💾 Store memories with rich metadata
- 🔍 Retrieve memories using semantic search
- 🧠 Automatic summarization and abstraction
- 🔄 Track changes in information over time
- 📈 Automatically track importance and relevance
- 📝 Support for structured and unstructured data
- 🚀 Built-in MCP-compatible server for easy integration
- 🧩 MCP compatible for agent ecosystems
- 🧪 Comprehensive test suite with mocks
- 🌟 Support for specialized embedding task types (Gemini)
- 🤖 LLM-powered content analysis and segmentation

## Specialized Embedding Task Types

A-Mem supports specialized embedding task types with the Gemini API, which improves retrieval performance for different content types:

### Currently supported task types:

- `RETRIEVAL_DOCUMENT`: Optimized for embedding documents to be retrieved later
- `RETRIEVAL_QUERY`: Optimized for embedding queries to search for documents
- `CODE_RETRIEVAL_DOCUMENT`: Optimized for embedding code snippets
- `CODE_RETRIEVAL_QUERY`: Optimized for embedding queries to search for code
- `QUESTION_ANSWERING`: Optimized for Q&A applications

The system automatically detects the content type (code, question, or general text) and applies the appropriate task type for embeddings.

### Example

We provide an example script to demonstrate the effectiveness of specialized task types:

```bash
# Set your Google API key first
export GOOGLE_API_KEY=your_api_key_here
python examples/gemini_task_types_example.py
```

This example compares embeddings generated with different task types and shows how they affect search results.

## LLM-Powered Content Analysis and Segmentation

A-Mem uses LLMs to intelligently analyze and segment mixed content, providing more sophisticated handling than rule-based methods:

### Content Analysis

- **Advanced type detection**: Uses LLM to classify content with higher precision
- **Mixed content detection**: Identifies when content contains multiple types (code, documentation, questions)
- **Confidence scoring**: Provides confidence levels for content classification
- **Type proportions**: Analyzes the percentage breakdown of different content types
- **Optimal task type selection**: Dynamically determines the best embedding task type for each content piece

### Content Segmentation

- **Intelligent boundaries**: Identifies natural segment boundaries in mixed content
- **Coherent chunks**: Extracts semantically coherent parts while preserving syntactic structures
- **Segment metadata**: Generates rich metadata for each segment (type, subtitles, language detection)
- **Parent-child relationships**: Maintains connections between segments and original content
- **Individual optimization**: Each segment gets its optimal embedding task type

### Example

We provide an example script that demonstrates the advanced content analysis and segmentation:

```bash
# Set your API key first (works with both OpenAI and Google API keys)
export OPENAI_API_KEY=your_api_key_here
# or
export GOOGLE_API_KEY=your_api_key_here

python examples/llm_segmentation_example.py
```

This example shows how mixed content (like a tutorial with code snippets, explanations, and questions) gets analyzed and segmented, with each part stored and retrieved optimally.

## Server Implementation

A-Mem can be deployed as a server using the MCP interface:

```bash
python ./amem/server/mcp_fastmcp_server.py
```

The server was designed as an SSE server to make it easier to allow for flexibility of loading local models in memory. There are certainly ways to delegate this to external services and/or have the MCP server exist as a thin Stdio layer over the server - open to discussion and contributions. The server can also be run in Docker.

### Configuration

Configure the server using environment variables or command-line arguments:

```bash
# With environment variables
export PROJECT_NAME=my_project
export LLM_BACKEND=openai
export OPENAI_API_KEY=your_api_key
python mcp_fastmcp_server.py

# With command-line arguments
python mcp_fastmcp_server.py --project-name my_project --llm-backend openai --openai-api-key your_api_key
```

See the `.env.example` file for all available configuration options.


## TypeScript Client

The project includes a TypeScript client for connecting to the AMemCP server. The client uses the TS Model Context Protocol (MCP) SDK to communicate with the server. At present the client is mainly used for testing.

### Client Features

- Connect to the AMemCP server using SSE transport
- Create, retrieve, and search memories
- List available tools on the server
- Built with TypeScript for type safety

### Running the TypeScript Client

```bash
# Navigate to the client directory
cd client

# Install dependencies
npm install

# Build and run the client
npm run dev
```

For more details, see the [client README](./client/README.md).

## License

[MIT License](LICENSE)