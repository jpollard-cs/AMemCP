#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== AMemCP Setup Script ===${NC}"

# Check if Python 3.11 is available through pyenv
if ! pyenv versions | grep -q "3.11"; then
    echo -e "${YELLOW}Python 3.11 not found in pyenv. Installing latest Python 3.11.x...${NC}"
    pyenv install 3.11 --skip-existing
fi

# Get the latest installed 3.11.x version
PYTHON_VERSION=$(pyenv versions | grep -o "3.11.[0-9]*" | sort -V | tail -1)
if [ -z "$PYTHON_VERSION" ]; then
    echo -e "${RED}Failed to find Python 3.11.x. Please install it manually using pyenv.${NC}"
    exit 1
fi

echo -e "${GREEN}Using Python ${PYTHON_VERSION}${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${GREEN}Creating virtual environment in ./venv${NC}"
    $(pyenv root)/versions/${PYTHON_VERSION}/bin/python -m venv venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment${NC}"
source venv/bin/activate

# Check if activation worked
if [[ "$VIRTUAL_ENV" != *"venv"* ]]; then
    echo -e "${RED}Failed to activate virtual environment. Please run: source venv/bin/activate${NC}"
    exit 1
fi

# Verify Python version in the virtual environment
VENV_PYTHON_VERSION=$(python --version)
echo -e "${GREEN}Virtual environment is using: ${VENV_PYTHON_VERSION}${NC}"

# Upgrade pip, setuptools, and wheel
echo -e "${GREEN}Upgrading pip, setuptools, and wheel${NC}"
pip install --upgrade pip setuptools wheel

# Install requirements
echo -e "${GREEN}Installing requirements from requirements.txt${NC}"
pip install -r requirements.txt

# Install package in development mode
echo -e "${GREEN}Installing package in development mode${NC}"
pip install -e .

echo -e "${GREEN}=== Setup complete! ===${NC}"
echo -e "${YELLOW}To activate this environment in the future, run:${NC}"
echo -e "    source venv/bin/activate"
echo -e "${GREEN}To run the server:${NC}"
echo -e "    amem-server"

# Keep the environment activated for the current shell
echo -e "${GREEN}Virtual environment is now active. You can start developing!${NC}"
