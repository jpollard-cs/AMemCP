#!/usr/bin/env python3
"""Helper module for SSE transport in MCP server."""

import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Mount, Route

logger = logging.getLogger(__name__)


def create_sse_server(mcp: FastMCP):
    """Create a Starlette app that handles SSE connections and message handling."""
    # Initialize the SSE transport with the path to handle messages
    transport = SseServerTransport("/messages/")

    # Define the SSE connection handler
    async def handle_sse(request):
        """Handle incoming SSE connections."""
        # Log client information
        client_info = request.client if hasattr(request, "client") else "unknown"
        logger.info(f"New SSE connection from {client_info}")

        # Extract user ID from header (if provided)
        user_id = request.headers.get("user-id")
        if user_id:
            logger.info(f"Connection with User-Id: {user_id}")

        # Extract session ID if provided in query params (check both formats for compatibility)
        session_id = request.query_params.get("session_id") or request.query_params.get("sessionId")
        if session_id:
            logger.info(f"Connection with session_id: {session_id}")

        # Use the transport to handle the SSE connection
        async with transport.connect_sse(request.scope, request.receive, request._send) as streams:
            # Run the MCP server with the provided streams
            await mcp._mcp_server.run(streams[0], streams[1], mcp._mcp_server.create_initialization_options())

    # Create Starlette routes for SSE and message handling
    routes = [
        Route("/sse/", endpoint=handle_sse),
        Mount("/messages/", app=transport.handle_post_message),
    ]

    # Create a Starlette app with the defined routes
    return Starlette(routes=routes)
