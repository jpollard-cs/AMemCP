#!/usr/bin/env python3

import asyncio
import json
import logging
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

import httpx
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom MessageStream implementation since mcp.streams is not publicly available
class MessageStream:
    """A simple message stream to handle MCP communication."""

    def __init__(self, handler: Callable[[Any], Awaitable[Any]]):
        """Initialize with a handler function.

        Args:
            handler: Either a receive or send function for messages
        """
        self._handler = handler

    async def receive(self) -> Any:
        """Receive a message from the stream."""
        return await self._handler()

    async def send(self, message: Any) -> None:
        """Send a message to the stream."""
        return await self._handler(message)


# Simplified implementation to parse SSE messages
async def parse_sse_stream(response):
    """Parse Server-Sent Events from an HTTP response stream."""
    event_type = None
    data = ""

    async for line in response.aiter_lines():
        line = line.rstrip("\n")

        if not line:  # Empty line marks the end of an event
            if data:
                yield {"event": event_type or "message", "data": data}
            event_type = None
            data = ""
            continue

        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()

    # Yield any final event
    if data:
        yield {"event": event_type or "message", "data": data}


# Workaround for redirect handling issue in mcp.client.sse
# Based on PR: https://github.com/modelcontextprotocol/python-sdk/pull/284
@asynccontextmanager
async def patched_sse_client(url, headers=None):
    """Patched version of sse_client that handles redirects properly"""
    import anyio
    from mcp.client.sse import remove_request_params

    headers = headers or {}

    # Create task group for managing background tasks
    async with anyio.create_task_group() as tg:
        try:
            logger.info(f"Connecting to SSE endpoint: {remove_request_params(url)}")

            # Create client with redirect handling
            client = httpx.AsyncClient(headers=headers, follow_redirects=True)

            # Create memory object stream for passing messages between tasks
            # This creates a pair of objects - one for sending, one for receiving
            send_stream, receive_stream = anyio.create_memory_object_stream(100)

            try:
                # Use stream as a context manager
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    logger.debug(f"Connected to SSE endpoint, status: {response.status_code}")

                    # Task to receive SSE events
                    async def receive_sse_events():
                        try:
                            async for event in parse_sse_stream(response):
                                if event.get("event") == "message":
                                    data = event.get("data")
                                    if data:
                                        try:
                                            msg = json.loads(data)
                                            logger.debug(f"Received SSE message: {msg}")
                                            await send_stream.send(msg)
                                        except json.JSONDecodeError:
                                            logger.error(f"Failed to decode message: {data}")
                        except Exception as e:
                            logger.error(f"Error in receive_sse_events: {str(e)}")
                            if logging.getLogger().level <= logging.DEBUG:
                                traceback.print_exc()

                    # Start the background task
                    tg.start_soon(receive_sse_events)

                    # Get the first message (endpoint information)
                    try:
                        msg = await receive_stream.receive()
                        logger.debug(f"Received message: {msg}")

                        if msg.get("type") != "endpoint":
                            raise ValueError(f"Unexpected message type: {msg.get('type')}")

                        endpoint = msg.get("endpoint")
                        session_id = msg.get("session_id")
                        logger.debug(f"Received endpoint: {endpoint}, session_id: {session_id}")

                        # Function to send messages to the server
                        async def send_message(message):
                            payload = json.dumps(message)
                            async with httpx.AsyncClient(follow_redirects=True) as post_client:
                                post_response = await post_client.post(
                                    endpoint,
                                    content=payload,
                                    headers={"Content-Type": "application/json"},
                                )
                                post_response.raise_for_status()

                        # Create streams for client/server communication
                        read_stream = MessageStream(receive_stream.receive)
                        write_stream = MessageStream(send_message)

                        try:
                            # Yield the streams for use in the with statement
                            yield [read_stream, write_stream]
                        finally:
                            # Clean up when the context manager exits
                            await send_stream.aclose()
                            await receive_stream.aclose()
                    except Exception as e:
                        logger.error(f"Error processing SSE messages: {str(e)}")
                        if logging.getLogger().level <= logging.DEBUG:
                            traceback.print_exc()
                        raise
            finally:
                await client.aclose()
        except Exception as e:
            logger.error(f"Error in patched SSE client: {str(e)}")
            if logging.getLogger().level <= logging.DEBUG:
                traceback.print_exc()
            raise
        finally:
            # Make sure the task group gets canceled
            tg.cancel_scope.cancel()


async def test_memory_operations(session):
    """Test memory-related operations"""
    try:
        print("\n=== Testing Memory Operations ===")

        # Create entities
        print("\nCreating entities...")
        create_result = await session.execute_tool(
            "create_entities",
            {
                "entities": [
                    {
                        "name": "Example Entity",
                        "entityType": "concept",
                        "observations": ["This is an example entity for testing"],
                    }
                ]
            },
        )
        print(f"Create entities result: {json.dumps(create_result, indent=2)}")

        # Search for entities
        print("\nSearching for entities...")
        search_result = await session.execute_tool("search_nodes", {"query": "example"})
        print(f"Search result: {json.dumps(search_result, indent=2)}")

        # Add observation
        print("\nAdding observation...")
        add_result = await session.execute_tool(
            "add_observations",
            {
                "observations": [
                    {"entityName": "Example Entity", "contents": ["Adding a new observation via official MCP client"]}
                ]
            },
        )
        print(f"Add observation result: {json.dumps(add_result, indent=2)}")

        # Open specific node
        print("\nOpening specific node...")
        open_result = await session.execute_tool("open_nodes", {"names": ["Example Entity"]})
        print(f"Open node result: {json.dumps(open_result, indent=2)}")

        return open_result
    except Exception as e:
        print(f"Error in memory operations: {str(e)}")
        if logging.getLogger().level <= logging.DEBUG:
            traceback.print_exc()
        return None


async def main():
    """Initialize and run the MCP client with SSE transport"""
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP SSE Client")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--use-patched-client", action="store_true", help="Use patched SSE client with redirect handling"
    )
    args = parser.parse_args()

    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        # Enable httpx logging
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("httpcore").setLevel(logging.DEBUG)

    server_url = args.url
    sse_endpoint = f"{server_url}/"

    print(f"Connecting to MCP server at {sse_endpoint}")

    try:
        # Connect to the SSE endpoint using either patched or original client
        if args.use_patched_client:
            print("Using patched SSE client with redirect handling")
            async with patched_sse_client(sse_endpoint) as streams:
                print("SSE connection established")

                # Create a client session
                async with ClientSession(streams[0], streams[1]) as session:
                    # Initialize the session
                    await session.initialize()
                    print("Session initialized")

                    # Get available tools
                    tools = await session.list_tools()
                    tool_names = [tool["name"] for tool in tools]
                    print(f"Available tools: {json.dumps(tool_names, indent=2)}")

                    # Test memory operations if available
                    memory_tools = ["create_entities", "search_nodes", "add_observations", "open_nodes"]
                    if all(tool in tool_names for tool in memory_tools):
                        await test_memory_operations(session)
                    else:
                        print("\nMemory operations not available in this MCP server")
                        print(f"Available tools are: {tool_names}")
        else:
            # Use the standard SSE client
            async with sse_client(sse_endpoint) as streams:
                print("SSE connection established")

                # Create a client session
                async with ClientSession(streams[0], streams[1]) as session:
                    # Initialize the session
                    await session.initialize()
                    print("Session initialized")

                    # Get available tools
                    tools = await session.list_tools()
                    tool_names = [tool["name"] for tool in tools]
                    print(f"Available tools: {json.dumps(tool_names, indent=2)}")

                    # Test memory operations if available
                    memory_tools = ["create_entities", "search_nodes", "add_observations", "open_nodes"]
                    if all(tool in tool_names for tool in memory_tools):
                        await test_memory_operations(session)
                    else:
                        print("\nMemory operations not available in this MCP server")
                        print(f"Available tools are: {tool_names}")

    except ConnectionRefusedError:
        print(f"Error: Could not connect to server at {sse_endpoint}. Is the server running?")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            print("\nTraceback:")
            traceback.print_exc()
        sys.exit(1)

    print("\nClient finished successfully")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient shutdown by user")
