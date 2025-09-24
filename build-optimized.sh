#!/bin/bash
# build-optimized.sh - Script for optimized Docker builds

echo "🚀 Starting optimized Docker build for AI Architect Demo API server"
echo "This script uses Docker BuildKit for better caching and parallel builds"

# Enable BuildKit for better performance
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Show build cache info
echo "📊 Current Docker build cache:"
docker system df

echo "🔨 Building API server with optimizations..."
echo "   ✅ Multi-stage build for better layer caching"
echo "   ✅ BuildKit enabled for parallel builds"
echo "   ✅ Dependency layer cached separately from code"
echo "   ✅ Build context minimized with .dockerignore"

# Run the optimized build
time docker-compose build api-server --no-cache --parallel

echo "✨ Build complete! Check the time improvement."
echo "💡 For subsequent builds, use: docker-compose up api-server --build"