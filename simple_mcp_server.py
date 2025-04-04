#!/usr/bin/env python3
import json
import sys
import uuid
from typing import Any, Dict


class SimpleMcpServer:
    def __init__(self):
        self.session_id = None
        self.tools = [
            {
                "name": "echo",
                "description": "Echo back the input",
                "parameters": {"message": {"type": "string", "description": "Message to echo"}},
            },
            {"name": "debug_info", "description": "Return debug information about the server state", "parameters": {}},
        ]

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming message and return a response"""
        print(f"Received message: {json.dumps(message, indent=2)}", file=sys.stderr)

        # Check for sessionId
        if "sessionId" in message:
            self.session_id = message["sessionId"]
            print(f"Session ID set to: {self.session_id}", file=sys.stderr)

        # Basic JSON-RPC validation
        if "jsonrpc" not in message or message["jsonrpc"] != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": message.get("id", None),
                "error": {"code": -32600, "message": "Invalid Request: Not a valid JSON-RPC 2.0 request"},
            }

        if "method" not in message:
            return {
                "jsonrpc": "2.0",
                "id": message.get("id", None),
                "error": {"code": -32600, "message": "Invalid Request: Missing method"},
            }

        method = message["method"]
        params = message.get("params", {})
        request_id = message.get("id", str(uuid.uuid4()))

        # Handle methods
        if method == "list_tools":
            return {"jsonrpc": "2.0", "id": request_id, "result": self.tools}
        elif method == "execute_tool":
            tool_name = params.get("tool_name", "")
            arguments = params.get("arguments", {})

            if tool_name == "echo":
                return {"jsonrpc": "2.0", "id": request_id, "result": arguments.get("message", "No message provided")}
            elif tool_name == "debug_info":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "session_id": self.session_id,
                        "tools_count": len(self.tools),
                        "environment": {"sessionId": self.session_id, "params": params, "message": message},
                    },
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
                }
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }


def main():
    """Run as a stdio server"""
    print("Simple MCP Server - Starting...", file=sys.stderr)
    server = SimpleMcpServer()

    # Read from stdin, write to stdout
    try:
        for line in sys.stdin:
            try:
                message = json.loads(line)
                response = server.handle_message(message)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                print(f"Invalid JSON: {line}", file=sys.stderr)
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error: Invalid JSON"},
                }
                print(json.dumps(error_response), flush=True)
    except KeyboardInterrupt:
        print("Server shutting down", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
