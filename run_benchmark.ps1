#!/usr/bin/env pwsh
# Pose Estimation Model Benchmark Runner
# Benchmarks all YOLO pose models and generates comparison plots

param(
    [string]$Video = "data_hockey.mp4",
    [int]$Frames = 300,
    [string]$Device = "auto",
    [switch]$NoMemoryReset
)

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  POSE ESTIMATION MODEL BENCHMARK" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""

# Detect hardware
Write-Host ">>> Detecting hardware..." -ForegroundColor Yellow
try {
    $cpu = (Get-WmiObject -Class Win32_Processor | Select-Object -First 1).Name.Trim()
    $ramGB = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
    Write-Host "  CPU: $cpu" -ForegroundColor White
    Write-Host "  RAM: $ramGB GB" -ForegroundColor White

    # Try to get GPU
    $gpu = "CPU Only"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $nvidiaInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
        if ($nvidiaInfo) {
            $gpu = $nvidiaInfo.Trim()
        }
    }
    Write-Host "  GPU: $gpu" -ForegroundColor White
} catch {
    Write-Warning "Could not detect all hardware"
}

# Determine device
if ($Device -eq "auto") {
    try {
        $pythonCheck = python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")' 2>$null
        if ($pythonCheck) {
            $Device = $pythonCheck.Trim()
        } else {
            $Device = "cpu"
        }
    } catch {
        $Device = "cpu"
    }
}
Write-Host "  Device: $Device" -ForegroundColor Cyan
Write-Host ""

# Check files
if (-not (Test-Path $Video)) {
    Write-Host "ERROR: Video file not found: $Video" -ForegroundColor Red
    exit 1
}

$benchmarkScript = "benchmark\benchmark_pose_models.py"
if (-not (Test-Path $benchmarkScript)) {
    Write-Host "ERROR: Benchmark script not found: $benchmarkScript" -ForegroundColor Red
    exit 1
}

# Create output directory
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$deviceName = $Device.ToUpper()
$gpuShort = if ($gpu -match "RTX (\d+)") { "RTX$($matches[1])" }
            elseif ($gpu -match "GTX (\d+)") { "GTX$($matches[1])" }
            elseif ($gpu -match "NVIDIA") { "NVIDIA" }
            else { $deviceName }
$outputDir = "benchmark_results\run_${timestamp}_${gpuShort}"

Write-Host ">>> Output directory: $outputDir" -ForegroundColor Yellow
Write-Host ""

# Run benchmark
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  BENCHMARKING ALL YOLO MODELS" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ">>> Video: $Video" -ForegroundColor Yellow
Write-Host ">>> Frames: $Frames" -ForegroundColor Yellow
Write-Host ">>> Models: YOLO-N, YOLO-S, YOLO-M, YOLO-L, YOLO-X" -ForegroundColor Yellow
Write-Host ">>> Memory Reset: $(if ($NoMemoryReset) { 'Disabled' } else { 'Enabled' })" -ForegroundColor Yellow
Write-Host ""

$benchmarkCmd = "python `"$benchmarkScript`" --video `"$Video`" --frames $Frames --device $Device --yolo-sizes n s m l x --output-dir `"$outputDir`""

if ($NoMemoryReset) {
    $benchmarkCmd += " --no-memory-reset"
}

Write-Host "Running: $benchmarkCmd" -ForegroundColor Gray
Write-Host ""

try {
    Invoke-Expression $benchmarkCmd
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "SUCCESS: Benchmark completed!" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Benchmark failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
} catch {
    Write-Host "ERROR: Failed to run benchmark: $_" -ForegroundColor Red
    exit 1
}

# Process results
$plotFiles = Get-ChildItem -Path $outputDir -Filter "pose_benchmark_*.png" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($plotFiles.Count -gt 0) {
    $plotFile = $plotFiles[0].FullName
    Write-Host ""
    Write-Host "SUCCESS: Plot generated: $plotFile" -ForegroundColor Green

    # Add hardware info to filename (just use GPU name from folder)
    $hwString = $gpuShort
    $newPlotName = $plotFile -replace "pose_benchmark_", "pose_benchmark_${hwString}_"

    if ($plotFile -ne $newPlotName) {
        try {
            Move-Item -Path $plotFile -Destination $newPlotName -Force
            $plotFile = $newPlotName
            Write-Host "SUCCESS: Renamed with hardware info" -ForegroundColor Green
        } catch {
            Write-Warning "Could not rename plot file"
        }
    }

    # Open plot
    Write-Host ""
    Write-Host ">>> Opening plot..." -ForegroundColor Yellow
    try {
        Start-Process $plotFile
    } catch {
        Write-Warning "Could not open plot automatically"
    }
}

# Display summary
$jsonFiles = Get-ChildItem -Path $outputDir -Filter "benchmark_*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($jsonFiles.Count -gt 0) {
    $jsonFile = $jsonFiles[0].FullName
    Write-Host ""
    Write-Host "SUCCESS: Results saved: $jsonFile" -ForegroundColor Green

    try {
        $results = Get-Content $jsonFile | ConvertFrom-Json
        Write-Host ""
        Write-Host "================================================================" -ForegroundColor Cyan
        Write-Host "  BENCHMARK SUMMARY" -ForegroundColor Cyan
        Write-Host "================================================================" -ForegroundColor Cyan
        Write-Host ""

        Write-Host "Hardware Configuration:" -ForegroundColor Cyan
        Write-Host "  CPU: $cpu"
        Write-Host "  GPU: $gpu"
        Write-Host "  RAM: $ramGB GB"
        Write-Host "  Device: $deviceName"
        Write-Host ""

        Write-Host "Video Information:" -ForegroundColor Cyan
        Write-Host "  File: $($results.video_info.path)"
        Write-Host "  Resolution: $($results.video_info.resolution)"
        Write-Host "  FPS: $([math]::Round($results.video_info.fps, 2))"
        Write-Host "  Frames Processed: $($results.video_info.frames_processed)"
        Write-Host ""

        Write-Host "Model Performance:" -ForegroundColor Cyan
        Write-Host ""

        $tableFormat = "{0,-15} {1,8} {2,10} {3,10} {4,10}"
        Write-Host ($tableFormat -f "Model", "FPS", "Time(ms)", "Detect", "Conf") -ForegroundColor Yellow
        Write-Host ("-" * 63) -ForegroundColor Gray

        foreach ($model in $results.models.PSObject.Properties) {
            $name = $model.Name
            $data = $model.Value

            $fps = [math]::Round($data.mean_fps, 1)
            $time = [math]::Round($data.mean_inference_time_ms, 1)
            $detect = [math]::Round($data.mean_detections, 1)
            $conf = [math]::Round($data.mean_confidence, 3)

            Write-Host ($tableFormat -f $name, $fps, $time, $detect, $conf)
        }

        Write-Host ""

        # Find best models
        $models = $results.models.PSObject.Properties
        $fastestModel = $models | Sort-Object { $_.Value.mean_fps } -Descending | Select-Object -First 1
        $mostAccurate = $models | Sort-Object { $_.Value.mean_detections } -Descending | Select-Object -First 1

        Write-Host "Recommendations:" -ForegroundColor Cyan
        $fastestFPS = [math]::Round($fastestModel.Value.mean_fps, 1)
        $mostAccurateDet = [math]::Round($mostAccurate.Value.mean_detections, 1)
        Write-Host "  Fastest: $($fastestModel.Name) - $fastestFPS FPS"
        Write-Host "  Most Detections: $($mostAccurate.Name) - $mostAccurateDet players/frame"

        # Suggest balanced model
        $balancedModel = $models | Where-Object { $_.Name -like '*-M-*' } | Select-Object -First 1
        if ($balancedModel) {
            $balancedFPS = [math]::Round($balancedModel.Value.mean_fps, 1)
            $balancedDet = [math]::Round($balancedModel.Value.mean_detections, 1)
            Write-Host "  Balanced: $($balancedModel.Name) - $balancedFPS FPS, $balancedDet detect"
        }

    } catch {
        Write-Warning "Could not parse results JSON: $_"
    }
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  BENCHMARK COMPLETE" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "All results saved to: $outputDir" -ForegroundColor Green
Write-Host ""

# Offer to open results folder
$response = Read-Host "Open results folder? (Y/n)"
if ($response -eq "" -or $response -eq "y" -or $response -eq "Y") {
    Start-Process explorer.exe $outputDir
}

