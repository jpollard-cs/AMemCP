#!/usr/bin/env python3
"""
Simple MCP client test using the most basic approach.
"""

import asyncio
import logging

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    """Basic MCP client example"""
    try:
        # Default URLs based on how we're mounting in our server
        sse_url = "http://localhost:8000/mcp/sse"
        message_url = "http://localhost:8000/mcp/message"

        logger.info(f"Connecting to SSE endpoint at {sse_url}")
        logger.info(f"Using message endpoint at {message_url}")

        # Create the SSE client context manager
        sse_ctx = sse_client(sse_url, message_url)

        # Use async with for the context manager
        async with sse_ctx as streams:
            logger.info("SSE connection established")
            reader, writer = streams

            # Create an MCP client session
            session = ClientSession(reader, writer)

            # Initialize the session
            logger.info("Initializing MCP session...")
            init_options = await session.initialize()
            logger.info(f"MCP Session initialized: {init_options.server_name} {init_options.server_version}")

            # List available tools
            tools = await session.list_tools()
            logger.info(f"Available tools: {', '.join(tool.name for tool in tools)}")

            # Call a tool
            if "get_memory_stats" in {tool.name for tool in tools}:
                logger.info("Calling get_memory_stats...")
                stats = await session.call_tool("get_memory_stats", {})
                logger.info(f"Memory stats: {stats}")

    except Exception as e:
        logger.error(f"Error in MCP client: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
