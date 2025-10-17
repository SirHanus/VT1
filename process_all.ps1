# powershell
param(
    [string]$Folder = ".\videos_all",
    [string]$Script = ".\ultralytics_video\sam_offline.py",
    [string]$Python = "python",
    [string]$OutputFolder = ".\videos_all_processed"
)

# Ensure output folder exists
if (-not (Test-Path -Path $OutputFolder)) {
    New-Item -ItemType Directory -Path $OutputFolder | Out-Null
}

# Ensure input folder exists
if (-not (Test-Path -Path $Folder)) {
    Write-Host "Input folder not found: $Folder"
    exit 1
}

# Resolve absolute input/output paths for later relative-path calculations
$folderPath = (Resolve-Path -LiteralPath $Folder).Path
$outputRoot = (Resolve-Path -LiteralPath $OutputFolder).Path

# Find mp4 files recursively
$files = Get-ChildItem -Path $Folder -Filter *.mp4 -File -Recurse
if ($files.Count -eq 0) {
    Write-Host "No mp4 files found in $Folder (recursively)"
    exit 0
}

# Collect per-run metrics for aggregation
$allMetrics = @()

foreach ($f in $files) {
    $src = $f.FullName

    # Compute relative path under the input folder and build corresponding output directory
    $rel = $src.Substring($folderPath.Length)
    if ($rel.StartsWith('\\') -or $rel.StartsWith('/')) { $rel = $rel.TrimStart('\\','/') }

    $relDir = Split-Path $rel -Parent
    if ([string]::IsNullOrEmpty($relDir)) {
        $destDir = $outputRoot
    } else {
        $destDir = Join-Path -Path $outputRoot -ChildPath $relDir
    }

    # Ensure destination subfolder exists
    if (-not (Test-Path -Path $destDir)) {
        New-Item -ItemType Directory -Path $destDir -Force | Out-Null
    }

    # Build output paths
    $outMp4 = Join-Path -Path $destDir -ChildPath $f.Name
    $metricsJson = [System.IO.Path]::ChangeExtension($outMp4, ".json")

    Write-Host "Processing: $src -> $outMp4"
    & $Python $Script --source $src --out $outMp4 --metrics-json $metricsJson --max-frames 800 --sam-every 1 --sam-reinit 70
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: processing failed for $src (exit code $LASTEXITCODE)"
    }

    # Append metrics if present
    if (Test-Path -LiteralPath $metricsJson) {
        try {
            $m = Get-Content -LiteralPath $metricsJson -Raw | ConvertFrom-Json
            if ($m -ne $null) { $allMetrics += $m }
        } catch {
            Write-Host "Warning: failed to read metrics JSON for ${src}: $($_.Exception.Message)"
        }
    } else {
        Write-Host "Note: metrics JSON not found for $src"
    }
}

# Write aggregate metrics JSON
$aggregatePath = Join-Path -Path $outputRoot -ChildPath "metrics_aggregate.json"
try {
    $allMetrics | ConvertTo-Json -Depth 8 | Out-File -FilePath $aggregatePath -Encoding utf8
    Write-Host "Aggregate metrics written to: $aggregatePath"
} catch {
    Write-Host "Warning: failed to write aggregate metrics: $($_.Exception.Message)"
}
