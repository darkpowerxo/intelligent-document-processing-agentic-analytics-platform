# build-optimized.ps1 - PowerShell script for optimized Docker builds

Write-Host "🚀 Starting optimized Docker build for AI Architect Demo API server" -ForegroundColor Green
Write-Host "This script uses Docker BuildKit for better caching and parallel builds" -ForegroundColor Cyan

# Enable BuildKit for better performance
$env:DOCKER_BUILDKIT = 1
$env:COMPOSE_DOCKER_CLI_BUILD = 1

# Show build cache info
Write-Host "📊 Current Docker build cache:" -ForegroundColor Yellow
docker system df

Write-Host "🔨 Building API server with optimizations..." -ForegroundColor Green
Write-Host "   ✅ Multi-stage build for better layer caching" -ForegroundColor White
Write-Host "   ✅ BuildKit enabled for parallel builds" -ForegroundColor White
Write-Host "   ✅ Dependency layer cached separately from code" -ForegroundColor White
Write-Host "   ✅ Build context minimized with .dockerignore" -ForegroundColor White

# Run the optimized build with timing
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
docker-compose build api-server --parallel
$stopwatch.Stop()

Write-Host "✨ Build complete in $($stopwatch.Elapsed.TotalMinutes.ToString('N2')) minutes!" -ForegroundColor Green
Write-Host "💡 For subsequent builds, use: docker-compose up api-server --build" -ForegroundColor Cyan