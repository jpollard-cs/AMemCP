#!/usr/bin/env python3

import logging
import sys
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages entities, observations, and relations in memory"""

    def __init__(self):
        self.entities = {}
        self.relations = []

    async def create_entities(self, entities_data):
        """Create entities and store them in memory"""
        created = []
        for entity in entities_data:
            name = entity["name"]
            self.entities[name] = {
                "name": name,
                "entityType": entity["entityType"],
                "observations": entity.get("observations", []),
            }
            created.append(name)
            logger.info(f"Created entity: {name}")
        return {"created": created}

    async def search_nodes(self, query):
        """Search for entities by name or observations"""
        results = []
        query = query.lower()
        for name, entity in self.entities.items():
            if query in name.lower() or any(query in obs.lower() for obs in entity["observations"]):
                results.append(entity)
        logger.info(f"Search results for '{query}': {len(results)} entities")
        return {"results": results}

    async def add_observations(self, observations_data):
        """Add observations to existing entities"""
        updated = []
        for item in observations_data:
            name = item["entityName"]
            if name in self.entities:
                self.entities[name]["observations"].extend(item["contents"])
                updated.append(name)
                logger.info(f"Added observations to: {name}")
        return {"updated": updated}

    async def open_nodes(self, names):
        """Retrieve specific entities by name"""
        results = []
        for name in names:
            if name in self.entities:
                results.append(self.entities[name])
                logger.info(f"Opened node: {name}")
        return {"results": results}

    async def read_graph(self):
        """Read the entire graph"""
        logger.info("Reading entire graph")
        return {"entities": list(self.entities.values()), "relations": self.relations}


# Create memory manager
mem_manager = MemoryManager()


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """Manage server startup and shutdown lifecycle."""
    logger.info("Initializing server resources")
    try:
        yield {"memory_manager": mem_manager}
    finally:
        logger.info("Cleaning up server resources")


# Initialize MCP server with lifespan
server = Server("a-mem-server", "1.0.0", lifespan=server_lifespan)


# Define tool functions using call_tool
@server.call_tool()
async def create_entities(name: str, params: dict) -> dict:
    logger.info(f"Creating entities: {params}")
    ctx = server.request_context
    memory = ctx.lifespan_context["memory_manager"]
    return await memory.create_entities(params["entities"])


@server.call_tool()
async def search_nodes(name: str, params: dict) -> dict:
    logger.info(f"Searching for: {params}")
    ctx = server.request_context
    memory = ctx.lifespan_context["memory_manager"]
    return await memory.search_nodes(params["query"])


@server.call_tool()
async def add_observations(name: str, params: dict) -> dict:
    logger.info(f"Adding observations: {params}")
    ctx = server.request_context
    memory = ctx.lifespan_context["memory_manager"]
    return await memory.add_observations(params["observations"])


@server.call_tool()
async def open_nodes(name: str, params: dict) -> dict:
    logger.info(f"Opening nodes: {params}")
    ctx = server.request_context
    memory = ctx.lifespan_context["memory_manager"]
    return await memory.open_nodes(params["names"])


@server.call_tool()
async def read_graph(name: str, params: dict) -> dict:
    logger.info("Reading entire graph")
    ctx = server.request_context
    memory = ctx.lifespan_context["memory_manager"]
    return await memory.read_graph()


# Initialize SSE transport
sse_transport = SseServerTransport("/message")


# Define SSE handler using the Request object
async def handle_sse(request: Request):
    """Handle SSE connections"""
    logger.info("New SSE connection")
    try:
        async with sse_transport.connect_sse(
            request.scope,
            request.receive,
            request._send,  # type: ignore[reportPrivateUsage]
        ) as streams:
            logger.info("SSE connection established")
            init_options = server.create_initialization_options()
            await server.run(streams[0], streams[1], init_options)
    except Exception as e:
        logger.error(f"Error in SSE connection: {str(e)}")
        if logger.level <= logging.DEBUG:
            traceback.print_exc()
        raise


# Create Starlette app with routes
app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/message", app=sse_transport.handle_post_message),
    ]
)

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP SSE Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("mcp").setLevel(logging.DEBUG)
        logging.getLogger("uvicorn").setLevel(logging.INFO)

    try:
        # Start the server
        host = args.host
        port = args.port
        logger.info(f"Starting MCP SSE server on {host}:{port}")

        # Run the Starlette app with uvicorn
        uvicorn.run(app, host=host, port=port, log_level="debug" if args.verbose else "info")
    except KeyboardInterrupt:
        print("\nServer shutdown by user")
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)
