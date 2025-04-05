#!/usr/bin/env python3
"""Setup script for the AMemCP package."""

from setuptools import find_packages, setup

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Filter out comments and empty lines
requirements = [line for line in requirements if line and not line.startswith("#")]

setup(
    name="amem",
    version="0.1.0",
    description="Agentic Memory System - A memory system for agentic applications",
    author="",
    author_email="",
    url="",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "amem-server=run_server:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
