# Mock modules for testing
import sys


class MockLLM:
    def get_completion(self, prompt, response_format=None):
        import json

        if response_format and hasattr(response_format, "get") and response_format.get("type") == "json_schema":
            return json.dumps(
                {
                    "keywords": ["test", "mock", "memory"],
                    "context": "Test content for mocking memory operations",
                    "tags": ["test", "mock"],
                }
            )
        return "Mock LLM response"


class MockLLMController:
    def __init__(self, backend="mock", model="mock", api_key=None):
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.llm = MockLLM()


# Mock litellm
sys.modules["litellm"] = type("MockLitellm", (), {"completion": lambda *args, **kwargs: None})

# Override imports in memory_system and llm_controller
sys.modules["llm_controller"] = type(
    "MockLLMControllerModule", (), {"LLMController": MockLLMController, "__file__": "llm_controller.py"}
)


# Create mock for AgenticMemorySystem
class MockMemoryNote:
    def __init__(
        self,
        content,
        id=None,
        keywords=None,
        links=None,
        retrieval_count=None,
        timestamp=None,
        last_accessed=None,
        context=None,
        evolution_history=None,
        category=None,
        tags=None,
    ):
        import uuid

        self.content = content
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        self.timestamp = timestamp or "202505141200"
        self.last_accessed = last_accessed or "202505141200"
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []


class MockRetriever:
    def __init__(self, model_name="mock"):
        self.model_name = model_name

    def search(self, query, k=5):
        return [
            {"id": "mock-id-1", "content": "Mock result 1", "score": 0.95},
            {"id": "mock-id-2", "content": "Mock result 2", "score": 0.85},
        ]


class MockChromaRetriever:
    def __init__(self, collection_name=None):
        self.collection_name = collection_name or "mock_collection"
        self._client = type("MockClient", (), {"delete_collection": lambda name: None})


class MockAgenticMemorySystem:
    def __init__(
        self,
        model_name="mock",
        llm_backend="mock",
        llm_model="mock",
        api_key=None,
        persist_directory=None,
        collection_name=None,
    ):
        self.memories = {}
        self.retriever = MockRetriever(model_name)
        self.chroma_retriever = MockChromaRetriever(collection_name)
        self.llm_controller = MockLLMController(llm_backend, llm_model, api_key)

    def create(self, content, tags=None, category=None, timestamp=None):
        memory_id = str(uuid.uuid4())
        memory = MockMemoryNote(content=content, id=memory_id, tags=tags, category=category, timestamp=timestamp)
        self.memories[memory_id] = memory
        return memory_id

    def read(self, memory_id):
        if memory_id in self.memories:
            return self.memories[memory_id]
        raise KeyError(f"Memory with ID {memory_id} not found")

    def update(self, memory_id, content):
        if memory_id not in self.memories:
            raise KeyError(f"Memory with ID {memory_id} not found")
        self.memories[memory_id].content = content

    def delete(self, memory_id):
        if memory_id not in self.memories:
            raise KeyError(f"Memory with ID {memory_id} not found")
        del self.memories[memory_id]

    def search(self, query, k=5):
        results = []
        for memory_id, memory in self.memories.items():
            if query.lower() in memory.content.lower():
                results.append(
                    {
                        "id": memory_id,
                        "content": memory.content,
                        "score": 0.9,
                    }
                )
        # Add mock results if needed
        if not results or len(results) < k:
            for i in range(k - len(results)):
                results.append(
                    {
                        "id": f"mock-id-{i}",
                        "content": f"Mock content {i} for query: {query}",
                        "score": 0.8 - (i * 0.1),
                    }
                )
        return results[:k]


sys.modules["memory_system"] = type(
    "MockMemorySystemModule",
    (),
    {"AgenticMemorySystem": MockAgenticMemorySystem, "MemoryNote": MockMemoryNote, "__file__": "memory_system.py"},
)
#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from memory_system import AgenticMemorySystem


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="A-Mem MCP Server")
    parser.add_argument("--project", "-p", type=str, help="Project name for memory partitioning")
    parser.add_argument("--model", "-m", type=str, help="Embedding model name")
    parser.add_argument("--backend", "-b", type=str, help="LLM backend (openai/ollama)")
    parser.add_argument("--llm-model", "-l", type=str, help="LLM model name")
    parser.add_argument("--host", type=str, help="Host address to bind server to")
    parser.add_argument("--port", type=int, help="Port to run server on")

    # If no args provided but MCP CLI was used, parse MCP's args format
    if len(sys.argv) == 1 and "MCP_ARGS" in os.environ:
        args = parser.parse_args(os.environ["MCP_ARGS"].split())
    else:
        args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get project name from command line, environment variable, or default
    project_name = args.project or os.environ.get("MCP_PROJECT_NAME", "default")

    # Get server host/port for Docker compatibility
    # In Docker, we need to bind to 0.0.0.0 to be accessible from outside the container
    host = args.host or os.environ.get("MCP_HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("MCP_PORT", "8000"))

    # Initialize MCP server
    mcp = FastMCP(f"A-Mem MCP Server - {project_name}")

    # Get embedding model
    embedding_model = args.model or os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Get persist directory from environment variable or default
    persist_directory = os.environ.get("PERSIST_DIRECTORY", "chroma_db")

    # Initialize memory system with project-specific configuration
    # Use project name to partition ChromaDB collections
    memory_system = AgenticMemorySystem(
        model_name=embedding_model,
        llm_backend=args.backend or os.environ.get("LLM_BACKEND", "openai"),
        llm_model=args.llm_model or os.environ.get("LLM_MODEL", "gpt-4"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        # Set up persistent storage path with project name
        persist_directory=persist_directory,
        collection_name=f"amem_{project_name.replace(' ', '_').lower()}",
    )

    print(f"Starting A-Mem MCP Server for project: {project_name}")
    print(f"Server running at: http://{host}:{port}")
    print(f"Embedding model: {memory_system.retriever.model_name}")
    print(f"LLM backend: {memory_system.llm_controller.backend}")
    print(f"LLM model: {memory_system.llm_controller.model}")
    print(f"Collection: {memory_system.chroma_retriever.collection_name}")
    print(f"Persistence directory: {persist_directory}")

    # Memory creation tool
    @mcp.tool()
    def create_memory(
        content: str, tags: Optional[List[str]] = None, category: Optional[str] = None, timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new memory in the A-Mem system.

        Args:
            content: The main text content of the memory
            tags: Optional list of classification tags
            category: Optional classification category
            timestamp: Optional creation time in format YYYYMMDDHHMM

        Returns:
            Dictionary containing the created memory ID and metadata
        """
        try:
            memory_id = memory_system.create(content=content, tags=tags, category=category, timestamp=timestamp)
            memory = memory_system.read(memory_id)

            return {
                "status": "success",
                "memory_id": memory_id,
                "metadata": {
                    "content": memory.content,
                    "tags": memory.tags,
                    "keywords": memory.keywords,
                    "context": memory.context,
                    "category": memory.category,
                    "timestamp": memory.timestamp,
                },
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Memory retrieval tool
    @mcp.tool()
    def read_memory(memory_id: str) -> Dict[str, Any]:
        """
        Retrieve a memory by its ID.

        Args:
            memory_id: The unique identifier of the memory to retrieve

        Returns:
            Dictionary containing the memory content and metadata
        """
        try:
            memory = memory_system.read(memory_id)
            return {
                "status": "success",
                "memory_id": memory_id,
                "content": memory.content,
                "metadata": {
                    "tags": memory.tags,
                    "keywords": memory.keywords,
                    "context": memory.context,
                    "category": memory.category,
                    "timestamp": memory.timestamp,
                    "last_accessed": memory.last_accessed,
                    "retrieval_count": memory.retrieval_count,
                    "links": memory.links,
                },
            }
        except KeyError:
            return {"status": "error", "message": f"Memory with ID {memory_id} not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Memory search tool
    @mcp.tool()
    def search_memories(query: str, k: int = 5) -> Dict[str, Any]:
        """
        Search for memories based on a query.

        Args:
            query: The search query
            k: Maximum number of results to return

        Returns:
            Dictionary containing the search results
        """
        try:
            results = memory_system.search(query, k=k)
            return {"status": "success", "count": len(results), "results": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Memory update tool
    @mcp.tool()
    def update_memory(memory_id: str, content: str) -> Dict[str, Any]:
        """
        Update an existing memory's content.

        Args:
            memory_id: The unique identifier of the memory to update
            content: The new content for the memory

        Returns:
            Dictionary indicating success or failure
        """
        try:
            memory_system.update(memory_id, content)
            updated_memory = memory_system.read(memory_id)
            return {
                "status": "success",
                "memory_id": memory_id,
                "updated_content": updated_memory.content,
                "updated_metadata": {
                    "tags": updated_memory.tags,
                    "keywords": updated_memory.keywords,
                    "context": updated_memory.context,
                },
            }
        except KeyError:
            return {"status": "error", "message": f"Memory with ID {memory_id} not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Memory deletion tool
    @mcp.tool()
    def delete_memory(memory_id: str) -> Dict[str, Any]:
        """
        Delete a memory by its ID.

        Args:
            memory_id: The unique identifier of the memory to delete

        Returns:
            Dictionary indicating success or failure
        """
        try:
            memory_system.delete(memory_id)
            return {"status": "success", "message": f"Memory with ID {memory_id} was deleted"}
        except KeyError:
            return {"status": "error", "message": f"Memory with ID {memory_id} not found"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Get all memories resource
    @mcp.resource("memories://all")
    def get_all_memories() -> str:
        """
        Get a list of all memories in the system.

        Returns:
            JSON string containing all memories
        """
        try:
            memories_data = []
            for memory_id, memory in memory_system.memories.items():
                memories_data.append(
                    {
                        "id": memory_id,
                        "content": memory.content,
                        "tags": memory.tags,
                        "keywords": memory.keywords,
                        "context": memory.context,
                        "category": memory.category,
                        "timestamp": memory.timestamp,
                    }
                )

            return json.dumps({"project": project_name, "count": len(memories_data), "memories": memories_data})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    # Get project info resource
    @mcp.resource("project://info")
    def get_project_info() -> str:
        """
        Get information about the current project.

        Returns:
            JSON string containing project information
        """
        return json.dumps(
            {
                "project_name": project_name,
                "memory_count": len(memory_system.memories),
                "embedding_model": memory_system.retriever.model_name,
                "llm_backend": memory_system.llm_controller.backend,
                "llm_model": memory_system.llm_controller.model,
                "collection_name": memory_system.chroma_retriever.collection_name,
                "persist_directory": persist_directory,
            }
        )

    # Add a quick create memory prompt
    @mcp.prompt()
    def create_memory_prompt(content: str) -> str:
        """Create a new memory with the given content"""
        return f"Please create a new memory with the following content: {content}"

    # Run the MCP server
    # In Docker, we need to bind to 0.0.0.0 to be accessible from outside the container
    mcp.run(host=host, port=port)


if __name__ == "__main__":
    main()
