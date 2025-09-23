#!/bin/bash

# Ollama initialization script

echo "Starting Ollama and pulling models..."

# Start Ollama server in background
ollama serve &

# Wait for Ollama to be ready
echo "Waiting for Ollama server to start..."
while ! curl -s http://localhost:11434/api/tags >/dev/null; do
    sleep 2
done

echo "Ollama server is ready!"

# Pull the required model
echo "Pulling llama3.1:latest model (this may take a while)..."
ollama pull llama3.1:latest

echo "Model setup complete!"

# Keep the container running
wait