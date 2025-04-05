#!/bin/bash
# Build and run script for the AMemCP TypeScript client

echo "Building and running AMemCP TypeScript client..."

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

# Build and run
npm run dev

echo "Done!"
