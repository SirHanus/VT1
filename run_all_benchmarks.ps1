#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Comprehensive benchmark runner for pose estimation models and full pipeline.

.DESCRIPTION
    Runs both YOLO model comparison benchmarks AND full pipeline benchmarks,
    generating all plots for paper analysis.

.PARAMETER Video
    Path to the video file to benchmark (default: data_hockey.mp4)

.PARAMETER Frames
    Number of frames to process (default: 300 for model comparison, 100 for pipeline)

.PARAMETER Device
    Device to use: cuda or cpu (default: auto-detect)

.PARAMETER ModelSizes
    YOLO model sizes to test for model comparison (default: n,s,m,l,x)

.PARAMETER PipelineModelSize
    YOLO model size to use for pipeline benchmark (default: m)

.PARAMETER SkipModelComparison
    Skip the model comparison benchmark (only run pipeline benchmark)

.PARAMETER SkipPipeline
    Skip the pipeline benchmark (only run model comparison)

.PARAMETER NoMemoryReset
    Disable memory reset between model runs

.EXAMPLE
    .\run_all_benchmarks.ps1
    Runs all benchmarks with default settings

.EXAMPLE
    .\run_all_benchmarks.ps1 -Frames 200 -ModelSizes m,l,x
    Custom frames and model selection

.EXAMPLE
    .\run_all_benchmarks.ps1 -SkipPipeline
    Only run model comparison benchmark
#>

param(
    [string]$Video = "data_hockey.mp4",
    [int]$Frames = 300,
    [string]$Device = "auto",
    [string[]]$ModelSizes = @("n", "s", "m", "l", "x"),
    [string]$PipelineModelSize = "m",
    [switch]$SkipModelComparison,
    [switch]$SkipPipeline,
    [switch]$NoMemoryReset
)

# Color output functions
function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Text)
    Write-Host ">>> $Text" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Text)
    Write-Host "SUCCESS: $Text" -ForegroundColor Green
}

function Write-Info {
    param([string]$Text)
    Write-Host "  $Text" -ForegroundColor White
}

# Detect hardware
function Get-HardwareInfo {
    $hwInfo = @{
        CPU = "Unknown"
        GPU = "CPU Only"
        RAM = "Unknown"
    }

    try {
        $cpu = (Get-WmiObject -Class Win32_Processor | Select-Object -First 1).Name.Trim()
        $hwInfo.CPU = $cpu

        $ramGB = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
        $hwInfo.RAM = "$ramGB GB"

        if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
            $nvidiaInfo = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
            if ($nvidiaInfo) {
                $hwInfo.GPU = $nvidiaInfo.Trim()
            }
        }
    } catch {
        Write-Warning "Could not detect all hardware"
    }

    return $hwInfo
}

# Main script
Write-Header "COMPREHENSIVE POSE ESTIMATION BENCHMARKS"

# Detect hardware
Write-Step "Detecting hardware..."
$hw = Get-HardwareInfo
Write-Info "CPU: $($hw.CPU)"
Write-Info "GPU: $($hw.GPU)"
Write-Info "RAM: $($hw.RAM)"
Write-Host ""

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
Write-Info "Device: $Device"
Write-Host ""

# Check video exists
if (-not (Test-Path $Video)) {
    Write-Host "ERROR: Video file not found: $Video" -ForegroundColor Red
    exit 1
}

# Create timestamp for this run
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$gpuShort = if ($hw.GPU -match "RTX (\d+)") { "RTX$($matches[1])" }
            elseif ($hw.GPU -match "GTX (\d+)") { "GTX$($matches[1])" }
            elseif ($hw.GPU -match "NVIDIA") { "NVIDIA" }
            else { $Device.ToUpper() }

# Summary of what will be run
Write-Host "Benchmark Configuration:" -ForegroundColor Cyan
Write-Info "Video: $Video"
Write-Info "Frames: $Frames"
Write-Info "Device: $Device"
Write-Info "GPU: $gpuShort"
Write-Host ""

if (-not $SkipModelComparison) {
    Write-Info "Will run: Model Comparison ($($ModelSizes -join ', '))"
}
if (-not $SkipPipeline) {
    $pipelineFrames = [math]::Min($Frames, 100)
    Write-Info "Will run: Pipeline Benchmark ($PipelineModelSize) with $pipelineFrames frames"
}
Write-Host ""

$results = @{
    ModelComparison = $null
    Pipeline = $null
}

# =============================================================================
# 1. MODEL COMPARISON BENCHMARK
# =============================================================================
if (-not $SkipModelComparison) {
    Write-Header "BENCHMARK 1: YOLO MODEL COMPARISON"

    $benchmarkScript = "benchmark\benchmark_pose_models.py"
    if (-not (Test-Path $benchmarkScript)) {
        Write-Host "ERROR: Script not found: $benchmarkScript" -ForegroundColor Red
        exit 1
    }

    $outputDir = "benchmark_results\run_${timestamp}_${gpuShort}"
    Write-Step "Output: $outputDir"
    Write-Step "Models: $($ModelSizes -join ', ')"
    Write-Step "Memory Reset: $(if ($NoMemoryReset) { 'Disabled' } else { 'Enabled' })"
    Write-Host ""

    $modelSizesStr = $ModelSizes -join " "
    $benchmarkCmd = "python `"$benchmarkScript`" --video `"$Video`" --frames $Frames --device $Device --yolo-sizes $modelSizesStr --output-dir `"$outputDir`""

    if ($NoMemoryReset) {
        $benchmarkCmd += " --no-memory-reset"
    }

    Write-Host "Running: $benchmarkCmd" -ForegroundColor Gray
    Write-Host ""

    try {
        Invoke-Expression $benchmarkCmd
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Model comparison completed!"

            # Find and process results
            $plotFiles = Get-ChildItem -Path $outputDir -Filter "pose_benchmark_*.png" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
            if ($plotFiles.Count -gt 0) {
                $plotFile = $plotFiles[0].FullName

                # Rename with hardware info
                $newPlotName = $plotFile -replace "pose_benchmark_", "pose_benchmark_${gpuShort}_"
                if ($plotFile -ne $newPlotName) {
                    try {
                        Move-Item -Path $plotFile -Destination $newPlotName -Force
                        $plotFile = $newPlotName
                    } catch {
                        Write-Warning "Could not rename plot file"
                    }
                }

                $results.ModelComparison = $plotFile
                Write-Success "Plot 1: $plotFile"
            }

            # Display summary
            $jsonFiles = Get-ChildItem -Path $outputDir -Filter "benchmark_*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
            if ($jsonFiles.Count -gt 0) {
                $jsonFile = $jsonFiles[0].FullName
                try {
                    $data = Get-Content $jsonFile | ConvertFrom-Json
                    Write-Host ""
                    Write-Host "Model Comparison Summary:" -ForegroundColor Cyan
                    $tableFormat = "{0,-15} {1,8} {2,10} {3,10}"
                    Write-Host ($tableFormat -f "Model", "FPS", "Time(ms)", "Detect") -ForegroundColor Yellow
                    Write-Host ("-" * 50) -ForegroundColor Gray

                    foreach ($model in $data.models.PSObject.Properties) {
                        $name = $model.Name
                        $m = $model.Value
                        $fps = [math]::Round($m.mean_fps, 1)
                        $time = [math]::Round($m.mean_inference_time_ms, 1)
                        $detect = [math]::Round($m.mean_detections, 1)
                        Write-Host ($tableFormat -f $name, $fps, $time, $detect)
                    }
                } catch {
                    Write-Warning "Could not parse results"
                }
            }
        } else {
            Write-Host "ERROR: Model comparison failed with exit code $LASTEXITCODE" -ForegroundColor Red
        }
    } catch {
        Write-Host "ERROR: Failed to run model comparison: $_" -ForegroundColor Red
    }
}

# =============================================================================
# 2. PIPELINE BENCHMARK
# =============================================================================
if (-not $SkipPipeline) {
    Write-Header "BENCHMARK 2: FULL PIPELINE (YOLO + SAM2 + SigLIP)"

    $pipelineScript = "benchmark\benchmark_full_pipeline.py"
    if (-not (Test-Path $pipelineScript)) {
        Write-Host "ERROR: Script not found: $pipelineScript" -ForegroundColor Red
        exit 1
    }

    # Use fewer frames for pipeline (it's slower)
    $pipelineFrames = [math]::Min($Frames, 100)
    $pipelineOutputDir = "benchmark_results\pipeline\run_${timestamp}"

    Write-Step "Output: $pipelineOutputDir"
    Write-Step "Model: YOLO-$($PipelineModelSize.ToUpper())"
    Write-Step "Frames: $pipelineFrames (reduced for pipeline complexity)"
    Write-Step "Pipelines: YOLO-only, +SAM2, +SAM2+SigLIP"
    Write-Host ""

    $pipelineCmd = "python `"$pipelineScript`" --video `"$Video`" --frames $pipelineFrames --device $Device --model-size $PipelineModelSize --output-dir `"$pipelineOutputDir`""

    Write-Host "Running: $pipelineCmd" -ForegroundColor Gray
    Write-Host ""

    try {
        Invoke-Expression $pipelineCmd
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Pipeline benchmark completed!"

            # Find plot
            $plotFiles = Get-ChildItem -Path $pipelineOutputDir -Filter "pipeline_benchmark_*.png" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
            if ($plotFiles.Count -gt 0) {
                $plotFile = $plotFiles[0].FullName
                $results.Pipeline = $plotFile
                Write-Success "Plot 2: $plotFile"
            }

            # Display summary
            $jsonFiles = Get-ChildItem -Path $pipelineOutputDir -Filter "pipeline_benchmark_*.json" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
            if ($jsonFiles.Count -gt 0) {
                $jsonFile = $jsonFiles[0].FullName
                try {
                    $data = Get-Content $jsonFile | ConvertFrom-Json
                    Write-Host ""
                    Write-Host "Pipeline Comparison Summary:" -ForegroundColor Cyan
                    $tableFormat = "{0,-35} {1,10} {1,8} {3,12}"
                    Write-Host ($tableFormat -f "Pipeline", "Time(ms)", "FPS", "Overhead") -ForegroundColor Yellow
                    Write-Host ("-" * 70) -ForegroundColor Gray

                    $baseline = $null
                    foreach ($pipeline in $data.pipelines.PSObject.Properties) {
                        $name = $pipeline.Name
                        $p = $pipeline.Value
                        $time = [math]::Round($p.mean_inference_time_ms, 1)
                        $fps = [math]::Round($p.mean_fps, 1)

                        if ($null -eq $baseline) {
                            $baseline = $time
                            $overhead = "baseline"
                        } else {
                            $overheadPct = [math]::Round((($time - $baseline) / $baseline * 100), 0)
                            $overhead = "+$overheadPct%"
                        }

                        Write-Host ($tableFormat -f $name, $time, $fps, $overhead)
                    }
                } catch {
                    Write-Warning "Could not parse pipeline results"
                }
            }
        } else {
            Write-Host "ERROR: Pipeline benchmark failed with exit code $LASTEXITCODE" -ForegroundColor Red
        }
    } catch {
        Write-Host "ERROR: Failed to run pipeline benchmark: $_" -ForegroundColor Red
    }
}

# =============================================================================
# FINAL SUMMARY
# =============================================================================
Write-Header "BENCHMARK COMPLETE - SUMMARY"

Write-Host "Generated Plots for Paper:" -ForegroundColor Cyan
Write-Host ""

if ($results.ModelComparison) {
    Write-Host "1. Model Comparison Plot:" -ForegroundColor Yellow
    Write-Info "   File: $($results.ModelComparison)"
    Write-Info "   Shows: YOLO model performance comparison (FPS, inference time, detections)"
    Write-Info "   Use for: Justifying model selection in methodology section"
    Write-Host ""
}

if ($results.Pipeline) {
    Write-Host "2. Pipeline Benchmark Plot:" -ForegroundColor Yellow
    Write-Info "   File: $($results.Pipeline)"
    Write-Info "   Shows: Impact of SAM2 and SigLIP on pipeline performance"
    Write-Info "   Use for: Demonstrating full system overhead and bottlenecks"
    Write-Host ""
}

Write-Host "Hardware Configuration:" -ForegroundColor Cyan
Write-Info "CPU: $($hw.CPU)"
Write-Info "GPU: $($hw.GPU)"
Write-Info "RAM: $($hw.RAM)"
Write-Info "Device: $Device"
Write-Host ""

Write-Host "Next Steps for Paper:" -ForegroundColor Cyan
Write-Info "1. Review generated plots"
Write-Info "2. Select best visualizations"
Write-Info "3. Add to paper figures folder"
Write-Info "4. Reference hardware specs from this summary"
Write-Host ""

# Offer to open plots
if ($results.ModelComparison -or $results.Pipeline) {
    $response = Read-Host "Open generated plots? (Y/n)"
    if ($response -eq "" -or $response -eq "y" -or $response -eq "Y") {
        if ($results.ModelComparison) {
            Start-Process $results.ModelComparison
        }
        if ($results.Pipeline) {
            Start-Process $results.Pipeline
        }
    }

    Write-Host ""
    $response2 = Read-Host "Open results folders? (Y/n)"
    if ($response2 -eq "" -or $response2 -eq "y" -or $response2 -eq "Y") {
        if ($results.ModelComparison) {
            $folder = Split-Path $results.ModelComparison -Parent
            Start-Process explorer.exe $folder
        }
        if ($results.Pipeline) {
            $folder = Split-Path $results.Pipeline -Parent
            Start-Process explorer.exe $folder
        }
    }
}

Write-Host ""
Write-Host "All benchmarks complete!" -ForegroundColor Green
Write-Host ""

