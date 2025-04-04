#!/usr/bin/env python3
import argparse
import asyncio
import json
import sys

from supergateway_client import SupergatewaySseClient


async def test_memory_operations(client):
    """Test memory-related MCP operations"""
    print("\n=== Testing Memory Operations ===")

    # Create entities
    print("\nCreating entities...")
    create_result = await client.execute_tool(
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
    search_result = await client.execute_tool("search_nodes", {"query": "example"})
    print(f"Search result: {json.dumps(search_result, indent=2)}")

    # Add observation
    print("\nAdding observation...")
    add_result = await client.execute_tool(
        "add_observations",
        {"observations": [{"entityName": "Example Entity", "contents": ["Adding a new observation via Supergateway"]}]},
    )
    print(f"Add observation result: {json.dumps(add_result, indent=2)}")

    # Open specific node
    print("\nOpening specific node...")
    open_result = await client.execute_tool("open_nodes", {"names": ["Example Entity"]})
    print(f"Open node result: {json.dumps(open_result, indent=2)}")

    return open_result


async def test_file_operations(client):
    """Test file-related MCP operations"""
    print("\n=== Testing File Operations ===")

    # List current directory
    print("\nListing directory...")
    list_result = await client.execute_tool("list_dir", {"relative_workspace_path": "."})
    print(f"List directory result: {json.dumps(list_result, indent=2)}")

    # Try reading a file
    print("\nReading a file...")
    read_result = await client.execute_tool(
        "read_file", {"target_file": "supergateway_client.py", "should_read_entire_file": True}
    )
    # Only print the first few lines to avoid flooding the console
    if "result" in read_result and "content" in read_result["result"]:
        lines = read_result["result"]["content"].split("\n")
        summary = "\n".join(lines[:10]) + "\n... (truncated)"
        read_result["result"]["content"] = summary
    print(f"Read file result: {json.dumps(read_result, indent=2)}")

    return True


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Supergateway SSE client")
    parser.add_argument("--port", type=int, default=8000, help="Port for Supergateway (default: 8000)")
    parser.add_argument("--host", type=str, default="localhost", help="Host for Supergateway (default: localhost)")
    parser.add_argument(
        "--user-id", type=str, default="test-user-123", help="User ID for authentication (default: test-user-123)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Configuration
    base_url = f"http://{args.host}:{args.port}"
    headers = {"User-Agent": "SupergatewaySseClient/1.0"}

    print(f"Connecting to Supergateway at {base_url} with user_id={args.user_id}")

    # Initialize client with user_id
    client = SupergatewaySseClient(base_url=base_url, headers=headers, user_id=args.user_id)

    # Register event handlers
    async def handle_tool_output(params):
        print(f"\nTool output event: {json.dumps(params, indent=2)}")

    client.register_event_handler("tool_output", handle_tool_output)

    # Start the client
    print("Starting client...")
    await client.start(verbose=args.verbose)

    try:
        # Get available tools
        print("\nListing available tools...")
        tools = await client.list_tools()

        # Print tool names only to avoid flooding the console
        tool_names = [tool["name"] for tool in tools]
        print(f"Available tools: {json.dumps(tool_names, indent=2)}")

        # Test memory operations if available
        memory_tools = ["create_entities", "search_nodes", "add_observations", "open_nodes"]
        if all(tool in tool_names for tool in memory_tools):
            await test_memory_operations(client)
        else:
            print("\nMemory operations not available in this MCP server")

        # Test file operations if available
        file_tools = ["list_dir", "read_file"]
        if all(tool in tool_names for tool in file_tools):
            await test_file_operations(client)
        else:
            print("\nFile operations not available in this MCP server")

        # Get available resources
        print("\nListing available resources...")
        resources = await client.get_resources()
        print(f"Available resources: {json.dumps(resources, indent=2)}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
    finally:
        print("\nClosing client...")
        await client.close()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
