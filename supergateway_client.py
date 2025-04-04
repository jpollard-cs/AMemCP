import asyncio
import json
import queue
import sys
import time
import uuid
from threading import Thread
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
import requests


class SupergatewaySseClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        headers: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
    ):
        self.base_url = base_url
        self.sse_url = f"{base_url}/sse"
        self.message_url = f"{base_url}/message"
        self.session_id = str(uuid.uuid4())
        self.user_id = user_id or f"user-{self.session_id[:8]}"
        self.requests = {}
        self.response_handlers = {}
        self.event_handlers = {}
        self.default_headers = headers or {}
        self.default_headers.update(
            {
                "Content-Type": "application/json",
                "User-Id": self.user_id,
                # Pass session ID in header as well
                "X-Session-Id": self.session_id,
            }
        )
        self.client = httpx.AsyncClient(headers=self.default_headers)
        self.sse_response = None
        self.listening = False
        self.listen_thread = None
        self.event_queue = queue.Queue()
        self._tools_cache = None
        self.verbose = False

    async def start(self, verbose: bool = False):
        """Start the SSE listener in a background thread"""
        if self.listening:
            return

        self.verbose = verbose
        self.listening = True

        # Start a background thread for SSE
        self.listen_thread = Thread(target=self._listen_for_events_thread)
        self.listen_thread.daemon = True
        self.listen_thread.start()

        # Start a task to process events from the queue
        asyncio.create_task(self._process_events())

        # Wait for the connection to be established
        await asyncio.sleep(0.5)

        if self.verbose:
            print(
                f"Connected to {self.base_url} with user_id={self.user_id}, session_id={self.session_id}",
                file=sys.stderr,
            )

    def _listen_for_events_thread(self):
        """Thread function to listen for SSE events"""
        try:
            # Try multiple possible SSE endpoints
            endpoints = [
                f"{self.base_url}/?sessionId={self.session_id}",  # root with sessionId as query param
                f"{self.sse_url}?sessionId={self.session_id}",  # /sse with sessionId as query param
                self.base_url,  # root path only
                self.sse_url,  # /sse path only
            ]

            for sse_url in endpoints:
                if self.verbose:
                    print(f"Trying SSE connection to {sse_url}", file=sys.stderr)

                # Close any previous response
                if self.sse_response:
                    self.sse_response.close()

                # Use standard requests with stream=True for SSE
                try:
                    self.sse_response = requests.get(
                        sse_url,
                        headers=self.default_headers,
                        stream=True,
                        timeout=5.0,  # Short timeout for connection attempts
                    )

                    if self.sse_response.status_code == 200:
                        if self.verbose:
                            print(f"SSE connection established at {sse_url}", file=sys.stderr)
                        break
                except Exception as e:
                    if self.verbose:
                        print(f"Failed to connect to {sse_url}: {e}", file=sys.stderr)

            # Check if we got a successful connection
            if not self.sse_response or self.sse_response.status_code != 200:
                print(f"SSE connection failed on all endpoints", file=sys.stderr)
                self.listening = False
                return

            # Read the stream line by line
            for line in self.sse_response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")

                # Only process data: lines
                if line.startswith("data: "):
                    data = line[6:]  # Remove 'data: ' prefix

                    if self.verbose:
                        print(f"SSE raw data: {data}", file=sys.stderr)

                    try:
                        json_data = json.loads(data)
                        # Put the event in the queue for async processing
                        self.event_queue.put(json_data)
                    except json.JSONDecodeError:
                        print(f"Received invalid JSON: {data}", file=sys.stderr)

        except Exception as e:
            print(f"SSE connection error in thread: {e}", file=sys.stderr)
            self.listening = False

            # Try to reconnect after a delay
            time.sleep(5)
            if not self.listening:
                self._listen_for_events_thread()

    async def _process_events(self):
        """Async task to process events from the queue"""
        while self.listening:
            try:
                # Check the queue with a timeout to not block the event loop
                try:
                    json_data = self.event_queue.get(block=False)

                    if self.verbose:
                        print(f"← {json.dumps(json_data, indent=2)}", file=sys.stderr)

                    if "jsonrpc" in json_data:
                        await self.handle_jsonrpc_message(json_data)
                    else:
                        print(f"Received non-JSON-RPC message: {json_data}", file=sys.stderr)

                except queue.Empty:
                    # No events in queue, continue after a small delay
                    pass
            except Exception as e:
                print(f"Error processing SSE event: {e}", file=sys.stderr)

            # Small delay to avoid busy-waiting
            await asyncio.sleep(0.1)

    async def handle_jsonrpc_message(self, data: Dict[str, Any]):
        """Handle incoming JSON-RPC messages"""
        if "id" in data and data["id"] in self.requests:
            request_id = data["id"]
            if request_id in self.response_handlers:
                await self.response_handlers[request_id](data)
                del self.response_handlers[request_id]
        elif "method" in data:
            method = data["method"]
            if method in self.event_handlers:
                await self.event_handlers[method](data.get("params", {}))

    async def send_message(self, message: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Send a message to the server"""
        if "id" not in message:
            message["id"] = str(uuid.uuid4())

        if "jsonrpc" not in message:
            message["jsonrpc"] = "2.0"

        # Add sessionId to the query string, not in the message body
        message_url_with_session = f"{self.message_url}?sessionId={self.session_id}"

        request_id = message["id"]
        self.requests[request_id] = message

        response_future = asyncio.Future()

        async def handle_response(response):
            response_future.set_result(response)

        self.response_handlers[request_id] = handle_response

        if self.verbose:
            print(f"→ {json.dumps(message, indent=2)}", file=sys.stderr)
            print(f"URL: {message_url_with_session}", file=sys.stderr)

        try:
            headers = dict(self.default_headers)
            # Ensure Content-Type is set correctly for this request
            headers["Content-Type"] = "application/json"

            response = await self.client.post(message_url_with_session, json=message, headers=headers, timeout=timeout)

            # Handle synchronous response (not via SSE)
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if self.verbose:
                        print(f"Received synchronous response: {json.dumps(response_data, indent=2)}", file=sys.stderr)

                    # If we got a direct response, resolve the future with it
                    if "id" in response_data and response_data["id"] == request_id:
                        response_future.set_result(response_data)
                        return await response_future
                except (json.JSONDecodeError, ValueError):
                    # Not a JSON response, continue with async flow
                    pass

            if response.status_code != 200:
                error_msg = f"Error sending message: {response.status_code} {response.text}"
                print(error_msg, file=sys.stderr)
                response_future.set_exception(Exception(error_msg))
                return {"error": {"code": response.status_code, "message": response.text}}

            # Set a timeout for the response
            try:
                result = await asyncio.wait_for(response_future, timeout)
                return result
            except asyncio.TimeoutError:
                error_msg = f"Timeout waiting for response to message: {message}"
                print(error_msg, file=sys.stderr)
                if request_id in self.response_handlers:
                    del self.response_handlers[request_id]
                return {"error": {"code": -32000, "message": "Response timeout"}}

        except Exception as e:
            error_msg = f"Exception sending message: {str(e)}"
            print(error_msg, file=sys.stderr)
            return {"error": {"code": -32000, "message": str(e)}}

    def register_event_handler(self, method: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Register a handler for a specific event method"""
        self.event_handlers[method] = handler

    async def close(self):
        """Close the client connections"""
        self.listening = False

        # Close the SSE response if it exists
        if self.sse_response is not None:
            self.sse_response.close()

        # Wait for thread to finish
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=1.0)

        await self.client.aclose()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools from the MCP server"""
        if self._tools_cache is not None:
            return self._tools_cache

        response = await self.send_message({"method": "list_tools", "params": {}})

        if "result" in response:
            self._tools_cache = response["result"]
            return self._tools_cache
        else:
            print(f"Error listing tools: {response.get('error', 'Unknown error')}", file=sys.stderr)
            return []

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
        """Execute a tool on the MCP server"""
        # Make a copy of arguments to avoid modifying the original
        args_copy = dict(arguments)

        response = await self.send_message(
            {"method": "execute_tool", "params": {"tool_name": tool_name, "arguments": args_copy}}, timeout=timeout
        )

        return response

    async def get_resources(self) -> List[Dict[str, Any]]:
        """Get the list of available resources from the MCP server"""
        response = await self.send_message({"method": "list_resources", "params": {}})

        if "result" in response:
            return response["result"]
        else:
            print(f"Error listing resources: {response.get('error', 'Unknown error')}", file=sys.stderr)
            return []

    async def get_resource(self, resource_name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get a specific resource from the MCP server"""
        params = {"resource_name": resource_name}
        if args:
            params["arguments"] = args

        response = await self.send_message({"method": "get_resource", "params": params})

        return response


async def main():
    """Example usage of the SupergatewaySseClient"""
    # Configuration
    base_url = "http://localhost:8000"
    headers = {
        "User-Agent": "SupergatewaySseClient/1.0"
        # Add any authentication headers if needed
        # "Authorization": "Bearer your-token"
    }

    # Create client with user ID
    user_id = "test-user-123"
    client = SupergatewaySseClient(base_url=base_url, headers=headers, user_id=user_id)

    # Register event handlers
    async def handle_tool_output(params):
        print(f"Tool output: {json.dumps(params, indent=2)}")

    client.register_event_handler("tool_output", handle_tool_output)

    # Start listening for events with verbose output
    await client.start(verbose=True)

    try:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {json.dumps(tools, indent=2)}")

        if tools:
            # Example: Call the first tool (replace with your specific tool)
            tool_name = tools[0]["name"]
            tool_params = {}  # Add appropriate parameters based on the tool

            print(f"Executing tool: {tool_name}")
            result = await client.execute_tool(tool_name, tool_params)
            print(f"Tool execution result: {json.dumps(result, indent=2)}")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
