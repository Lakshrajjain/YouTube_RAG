#!/bin/bash

# Build script for the YouTube RAG Chrome extension

# Create build directory
mkdir -p dist

# Copy files
cp manifest.json dist/
cp popup.html dist/
cp popup.js dist/
cp content.js dist/
cp content.css dist/
cp background.js dist/

# Copy icons
mkdir -p dist/icons
cp icons/* dist/icons/

echo "Chrome extension built successfully in dist/ directory"
