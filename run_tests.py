#!/usr/bin/env python3
"""
Test runner for A-Mem
Runs test suites for different components
"""

import argparse
import os
import subprocess
import sys

# Add the parent directory to Python path for importing modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Define test suites with paths relative to the tests directory
TEST_SUITES = {
    "mock": ["tests/test_mock_memory.py"],
    "memory": ["tests/test_memory_system.py"],
    "fastmcp": ["tests/test_fastmcp_server.py"],
    "mcp": ["tests/test_mcp_server.py"],
    "mcpclient": ["tests/test_mcp_client.py"],
}

# Default tests to run
DEFAULT_TESTS = TEST_SUITES["mock"] + TEST_SUITES["fastmcp"]


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run tests for A-Mem")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--mock", action="store_true", help="Run mock tests")
    parser.add_argument("--memory", action="store_true", help="Run memory system tests")
    parser.add_argument("--fastmcp", action="store_true", help="Run FastMCP server tests")
    parser.add_argument("--mcp", action="store_true", help="Run MCP server tests")
    parser.add_argument("--mcpclient", action="store_true", help="Run MCP client tests against Docker container")
    parser.add_argument("--docker", action="store_true", help="Run tests against Docker container")
    return parser.parse_args()


def main():
    """Main function"""
    args = get_args()

    # Determine which tests to run
    tests_to_run = []

    if args.all:
        # Run all tests
        for suite in TEST_SUITES.values():
            tests_to_run.extend(suite)
    elif args.mock or args.memory or args.fastmcp or args.mcp or args.mcpclient:
        # Run specific test suites
        if args.mock:
            tests_to_run.extend(TEST_SUITES["mock"])
        if args.memory:
            tests_to_run.extend(TEST_SUITES["memory"])
        if args.fastmcp:
            tests_to_run.extend(TEST_SUITES["fastmcp"])
        if args.mcp:
            tests_to_run.extend(TEST_SUITES["mcp"])
        if args.mcpclient:
            tests_to_run.extend(TEST_SUITES["mcpclient"])
    else:
        # Run default tests
        tests_to_run = DEFAULT_TESTS

    # Special handling for Docker tests
    if args.docker:
        # Only run client tests against Docker
        tests_to_run = TEST_SUITES["mcpclient"]

        # Check if Docker container is running
        check_docker = subprocess.run(
            ["docker", "ps", "--filter", "name=amem-mcp-server", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )

        if "amem-mcp-server" not in check_docker.stdout:
            print("Docker container 'amem-mcp-server' is not running.")
            print("Start it with: docker-compose up -d")
            return 1

    # Run tests
    print(f"Running tests: {', '.join(tests_to_run)}")
    for test in tests_to_run:
        print(f"\n=== Running {test} ===")
        result = subprocess.run([sys.executable, test], check=False)
        if result.returncode != 0:
            print(f"Test {test} failed with exit code {result.returncode}")
            return result.returncode

    print("\nAll tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
