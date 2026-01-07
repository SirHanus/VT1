#!/usr/bin/env pwsh
# Comprehensive benchmark runner with virtual environment support

param(
    [string]$Video = "data_hockey_benchmark.mp4",
    [int]$Frames = 800,
    [string]$Device = "auto",
    [string[]]$ModelSizes = @("n", "s", "m", "l", "x"),
    [string]$PipelineModelSize = "m",
    [string]$Frameworks = "mediapipe,pytorch,yolo",
    [switch]$SkipModelComparison,
    [switch]$SkipPipeline,
    [switch]$SkipFrameworks,
    [bool]$ExportFrameworks = $true,
    [int]$NumSamples = 10,
    [switch]$NoMemoryReset
)

# Get Python from virtual environment
function Get-PythonCommand {
    $venvPaths = @(
        ".venv\Scripts\python.exe",
        "venv\Scripts\python.exe"
    )

    foreach ($path in $venvPaths) {
        if (Test-Path $path) {
            return (Resolve-Path $path).Path
        }
    }

    if ($env:CONDA_PREFIX) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path $condaPython) {
            return $condaPython
        }
    }

    return "python"
}

$pythonCmd = Get-PythonCommand
Write-Host "Using Python: $pythonCmd" -ForegroundColor Cyan

# Verify packages
Write-Host "Verifying packages..." -ForegroundColor Yellow
$packagesOk = $true
$requiredPackages = @(
    @{Name="torch"; Import="torch"},
    @{Name="ultralytics"; Import="ultralytics"},
    @{Name="Pillow"; Import="PIL"},
    @{Name="reportlab"; Import="reportlab"},
    @{Name="mediapipe"; Import="mediapipe"}
)

foreach ($pkg in $requiredPackages) {
    $null = & $pythonCmd -c "import $($pkg.Import)" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Missing: $($pkg.Name)" -ForegroundColor Red
        $packagesOk = $false
    } else {
        Write-Host "  Found: $($pkg.Name)" -ForegroundColor Green
    }
}

if (-not $packagesOk) {
    Write-Host "ERROR: Install missing packages with: pip install torch ultralytics Pillow reportlab mediapipe" -ForegroundColor Red
    exit 1
}

Write-Host "All packages verified!" -ForegroundColor Green
Write-Host ""

# Detect device
if ($Device -eq "auto") {
    $Device = & $pythonCmd -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>$null
    if (-not $Device) { $Device = "cpu" }
}

# Detect hardware
$hw = @{CPU="Unknown"; GPU="CPU Only"; RAM="Unknown"}
try {
    $hw.CPU = (Get-WmiObject -Class Win32_Processor | Select-Object -First 1).Name.Trim()
    $hw.RAM = "$([math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)) GB"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $gpu = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
        if ($gpu) { $hw.GPU = $gpu.Trim() }
    }
} catch {}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$gpuShort = if ($hw.GPU -match "RTX (\d+)") { "RTX$($matches[1])" }
            elseif ($hw.GPU -match "GTX (\d+)") { "GTX$($matches[1])" }
            else { "GPU" }

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  BENCHMARK SUITE" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "GPU: $($hw.GPU)" -ForegroundColor White
Write-Host "Device: $Device" -ForegroundColor White
Write-Host "Video: $Video" -ForegroundColor White
Write-Host "Frames: $Frames" -ForegroundColor White
Write-Host ""

$results = @{
    Frameworks = $null
    ModelComparison = $null
    Pipeline = $null
}

# Framework Comparison
if (-not $SkipFrameworks) {
    Write-Host "Running Framework Comparison..." -ForegroundColor Yellow
    $frameworkScript = "benchmark\benchmark_frameworks.py"
    $frameworkFrames = [math]::Min($Frames, 100)
    $frameworkOutputDir = "benchmark_results\frameworks\run_$timestamp"

    if (Test-Path $frameworkScript) {
        $args = @($frameworkScript, "--video", $Video, "--frames", $frameworkFrames, "--device", $Device, "--yolo-size", $PipelineModelSize, "--output-dir", $frameworkOutputDir, "--frameworks", $Frameworks)
        if ($ExportFrameworks) {
            $args += "--export-frames"
            $args += "--num-samples"
            $args += $NumSamples
        }
        & $pythonCmd $args
        $plotFiles = Get-ChildItem -Path $frameworkOutputDir -Filter "framework_comparison_*.png" -ErrorAction SilentlyContinue
        if ($plotFiles) { $results.Frameworks = $plotFiles[0].FullName }
    }
}

# Model Comparison
if (-not $SkipModelComparison) {
    Write-Host "Running Model Comparison..." -ForegroundColor Yellow
    $benchmarkScript = "benchmark\benchmark_pose_models.py"
    $outputDir = "benchmark_results\pose\run_$timestamp"

    if (Test-Path $benchmarkScript) {
        $args = @($benchmarkScript, "--video", $Video, "--frames", $Frames, "--device", $Device, "--yolo-sizes") + $ModelSizes + @("--output-dir", $outputDir)
        if ($NoMemoryReset) { $args += "--no-memory-reset" }
        & $pythonCmd $args

        $plotFiles = Get-ChildItem -Path $outputDir -Filter "pose_benchmark_*.png" -ErrorAction SilentlyContinue
        if ($plotFiles) { $results.ModelComparison = $plotFiles[0].FullName }
    }
}

# Pipeline Benchmark
if (-not $SkipPipeline) {
    Write-Host "Running Pipeline Benchmark..." -ForegroundColor Yellow
    $pipelineScript = "benchmark\benchmark_full_pipeline.py"
    $pipelineFrames = [math]::Min($Frames, 100)
    $pipelineOutputDir = "benchmark_results\pipeline\run_$timestamp"

    if (Test-Path $pipelineScript) {
        & $pythonCmd $pipelineScript --video $Video --frames $pipelineFrames --device $Device --model-size $PipelineModelSize --output-dir $pipelineOutputDir
        $plotFiles = Get-ChildItem -Path $pipelineOutputDir -Filter "pipeline_benchmark_*.png" -ErrorAction SilentlyContinue
        if ($plotFiles) { $results.Pipeline = $plotFiles[0].FullName }
    }
}

# Generate PDF
$plotFiles = @()
if ($results.Frameworks) { $plotFiles += $results.Frameworks }
if ($results.ModelComparison) { $plotFiles += $results.ModelComparison }
if ($results.Pipeline) { $plotFiles += $results.Pipeline }

if ($plotFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "Generating PDF..." -ForegroundColor Yellow

    $pdfPath = "benchmark_results\benchmark_summary_${timestamp}_${gpuShort}.pdf"
    $pdfScript = "benchmark\create_benchmark_pdf.py"

    if (Test-Path $pdfScript) {
        $args = @($pdfScript) + $plotFiles + @($pdfPath)
        & $pythonCmd $args

        if (Test-Path $pdfPath) {
            Write-Host "PDF created: $pdfPath" -ForegroundColor Green
            $response = Read-Host "Open PDF? (Y/n)"
            if ($response -eq "" -or $response -match "^[Yy]") {
                Start-Process $pdfPath
            }
        }
    } else {
        Write-Warning "PDF script not found. Individual plots available."
    }
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  COMPLETE" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
if ($results.Frameworks) { Write-Host "Framework plot: $($results.Frameworks)" }
if ($results.ModelComparison) { Write-Host "Model plot: $($results.ModelComparison)" }
if ($results.Pipeline) { Write-Host "Pipeline plot: $($results.Pipeline)" }
Write-Host ""

