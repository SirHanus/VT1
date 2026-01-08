#!/usr/bin/env pwsh
# Setup and start YOLO ML Backend for Label Studio

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  YOLO ML Backend Setup for Label Studio" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    $null = docker ps 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Docker is not running. Please start Docker Desktop." -ForegroundColor Red
        exit 1
    }
    Write-Host "[✓] Docker is running" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Docker is not installed or not in PATH." -ForegroundColor Red
    exit 1
}

# Check if compose.yml exists
if (-not (Test-Path "compose.yml")) {
    Write-Host "ERROR: compose.yml not found. Please run this script from the project root." -ForegroundColor Red
    exit 1
}
Write-Host "[✓] compose.yml found" -ForegroundColor Green

# Check if ml_backend directory exists
if (-not (Test-Path "ml_backend")) {
    Write-Host "ERROR: ml_backend directory not found." -ForegroundColor Red
    exit 1
}
Write-Host "[✓] ml_backend directory found" -ForegroundColor Green

# Create directories if they don't exist
$dirs = @("labelstudio_data", "models")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "[✓] Created $dir directory" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Building YOLO ML Backend Docker image..." -ForegroundColor Yellow
docker compose build yolo-backend

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to build Docker image." -ForegroundColor Red
    exit 1
}

Write-Host "[✓] Docker image built successfully" -ForegroundColor Green
Write-Host ""

Write-Host "Starting services..." -ForegroundColor Yellow
docker compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to start services." -ForegroundColor Red
    exit 1
}

Write-Host "[✓] Services started" -ForegroundColor Green
Write-Host ""

Write-Host "Waiting for services to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Check if services are running
$lsRunning = docker ps --filter "name=vt1-labelstudio" --format "{{.Names}}" 2>$null
$mlRunning = docker ps --filter "name=vt1-yolo-backend" --format "{{.Names}}" 2>$null

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  SERVICE STATUS" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

if ($lsRunning) {
    Write-Host "[✓] Label Studio: RUNNING" -ForegroundColor Green
} else {
    Write-Host "[✗] Label Studio: NOT RUNNING" -ForegroundColor Red
}

if ($mlRunning) {
    Write-Host "[✓] YOLO Backend: RUNNING" -ForegroundColor Green
} else {
    Write-Host "[✗] YOLO Backend: NOT RUNNING" -ForegroundColor Red
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  ACCESS INFORMATION" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Label Studio UI:  http://localhost:9001" -ForegroundColor White
Write-Host "  Username:       admin" -ForegroundColor Gray
Write-Host "  Password:       admin" -ForegroundColor Gray
Write-Host ""
Write-Host "YOLO ML Backend:  http://localhost:9090" -ForegroundColor White
Write-Host "  Internal URL:   http://yolo-backend:9090" -ForegroundColor Gray
Write-Host ""

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  SETUP INSTRUCTIONS" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "1. Open Label Studio: http://localhost:9001" -ForegroundColor White
Write-Host "2. Login with admin/admin" -ForegroundColor White
Write-Host "3. Create or open a project" -ForegroundColor White
Write-Host "4. Go to: Settings → Machine Learning" -ForegroundColor White
Write-Host "5. Click 'Add Model'" -ForegroundColor White
Write-Host "6. Enter URL: http://yolo-backend:9090" -ForegroundColor Yellow
Write-Host "7. Click 'Validate and Save'" -ForegroundColor White
Write-Host "8. Enable 'Use for interactive preannotations'" -ForegroundColor White
Write-Host ""

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  USEFUL COMMANDS" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "View logs:        docker compose logs -f yolo-backend" -ForegroundColor Gray
Write-Host "Restart backend:  docker compose restart yolo-backend" -ForegroundColor Gray
Write-Host "Stop all:         docker compose down" -ForegroundColor Gray
Write-Host "Rebuild:          docker compose up -d --build" -ForegroundColor Gray
Write-Host ""

$response = Read-Host "Open Label Studio in browser? (Y/n)"
if ($response -eq "" -or $response -match "^[Yy]") {
    Start-Process "http://localhost:9001"
}

Write-Host ""
Write-Host "Setup complete! 🎉" -ForegroundColor Green

