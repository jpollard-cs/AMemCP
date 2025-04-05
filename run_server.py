#!/usr/bin/env python3
"""
Run the AMemCP server using Uvicorn.
"""

import argparse
import os
import sys
from typing import Dict

# Make sure amem module is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn


def parse_args() -> Dict:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the AMemCP server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reloading for development")
    parser.add_argument("--project", type=str, default=None, help="Project name (default: from env or 'default')")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory (default: from env or './data')")
    parser.add_argument("--log-level", type=str, default="info", help="Log level (default: info)")

    return vars(parser.parse_args())


def setup_environment(args: Dict) -> None:
    """Set up environment variables based on command line args."""
    if args.get("project"):
        os.environ["PROJECT_NAME"] = args["project"]

    if args.get("data_dir"):
        os.environ["PERSIST_DIRECTORY"] = args["data_dir"]

    # Ensure required env variables have defaults
    if "PROJECT_NAME" not in os.environ:
        os.environ["PROJECT_NAME"] = "default"

    if "PERSIST_DIRECTORY" not in os.environ:
        os.environ["PERSIST_DIRECTORY"] = "./data"


def main() -> None:
    """Main entry point."""
    try:
        from amem.utils.utils import setup_logger

        logger = setup_logger(__name__)

        args = parse_args()
        setup_environment(args)

        host = args.get("host", "0.0.0.0")
        port = args.get("port", 8000)
        reload = args.get("reload", False)
        log_level = args.get("log_level", "info")

        logger.info(f"Starting AMemCP server on {host}:{port}")
        logger.info(f"Project: {os.environ.get('PROJECT_NAME')}")
        logger.info(f"Data directory: {os.environ.get('PERSIST_DIRECTORY')}")

        uvicorn.run(
            "amem.server:fastmcp_app",
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
        )
    except KeyboardInterrupt:
        print("Server shutdown requested. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error running server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
