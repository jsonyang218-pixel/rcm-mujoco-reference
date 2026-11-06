param(
    [switch]$NoRestart
)

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$StageRoot = Join-Path $ProjectRoot "runs\paper_stage1"
$StatusPath = Join-Path $StageRoot "stage1_status.json"
$MonitorStatusPath = Join-Path $StageRoot "monitor_status.json"
$MonitorLogPath = Join-Path $StageRoot "monitor.log"
$TrainManifest = Join-Path $StageRoot "train\dataset\manifest.json"
$ValManifest = Join-Path $StageRoot "val\dataset\manifest.json"
$TestManifest = Join-Path $StageRoot "test\dataset\manifest.json"
$TrainVerification = Join-Path $StageRoot "train\dataset_verification.json"
$ValVerification = Join-Path $StageRoot "val\dataset_verification.json"
$TestVerification = Join-Path $StageRoot "test\dataset_verification.json"
$TrainLog = Join-Path $StageRoot "logs\train.log"
$Python = "E:\30_RESOURCES\32_课程教程\医学\中科院\rcm_mujoco_control_limit\rcm_mujoco_reference\.venv\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "run_stage1_collection.py"

New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null

function Write-MonitorLog([string]$Message) {
    $line = "[{0}] {1}" -f (Get-Date).ToString("o"), $Message
    Add-Content -LiteralPath $MonitorLogPath -Value $line -Encoding utf8
}

$collectorProcesses = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "python" -and
            $_.CommandLine -match "run_stage1_collection\.py|collect_v9_dataset\.py"
        }
)
$running = $collectorProcesses.Count -gt 0

$stageState = "missing"
if (Test-Path -LiteralPath $StatusPath) {
    try {
        $stageState = (Get-Content -Raw -LiteralPath $StatusPath |
            ConvertFrom-Json).status
    } catch {
        $stageState = "unreadable"
    }
}

$completed = 0
$passes = 0
$failures = 0
$frames = 0
if (Test-Path -LiteralPath $TrainManifest) {
    try {
        $manifest = Get-Content -Raw -LiteralPath $TrainManifest | ConvertFrom-Json
        $completed = @($manifest.episodes).Count
        $passes = @($manifest.episodes | Where-Object { $_.result -eq "PASS" }).Count
        $failures = $completed - $passes
        $frames = [int]$manifest.total_frames
    } catch {
        Write-MonitorLog "manifest_read_error=$($_.Exception.Message)"
    }
}

function Get-SplitCounts([string]$ManifestPath) {
    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        return [pscustomobject]@{completed=0; passes=0; failures=0; frames=0}
    }
    try {
        $value = Get-Content -Raw -LiteralPath $ManifestPath | ConvertFrom-Json
        $items = @($value.episodes)
        $passCount = @($items | Where-Object { $_.result -eq "PASS" }).Count
        return [pscustomobject]@{
            completed = $items.Count
            passes = $passCount
            failures = $items.Count - $passCount
            frames = [int]$value.total_frames
        }
    } catch {
        Write-MonitorLog "manifest_read_error=$ManifestPath $($_.Exception.Message)"
        return [pscustomobject]@{completed=0; passes=0; failures=0; frames=0}
    }
}

function Get-VerificationResult([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return "missing"
    }
    try {
        return (Get-Content -Raw -LiteralPath $Path |
            ConvertFrom-Json).result
    } catch {
        return "unreadable"
    }
}

$valCounts = Get-SplitCounts $ValManifest
$testCounts = Get-SplitCounts $TestManifest
$trainVerificationResult = Get-VerificationResult $TrainVerification
$valVerificationResult = Get-VerificationResult $ValVerification
$testVerificationResult = Get-VerificationResult $TestVerification

$disk = Get-PSDrive -Name E
$freeGb = [math]::Round($disk.Free / 1GB, 2)
$lastProgress = ""
if (Test-Path -LiteralPath $TrainLog) {
    $lastProgress = ((Get-Content -LiteralPath $TrainLog -Tail 3) -join " | ")
}

$summary = [ordered]@{
    timestamp = (Get-Date).ToString("o")
    running = $running
    stage_status = $stageState
    completed_train_episodes = $completed
    train_passes = $passes
    train_failures = $failures
    total_frames = $frames
    val_completed = $valCounts.completed
    val_passes = $valCounts.passes
    val_failures = $valCounts.failures
    val_frames = $valCounts.frames
    test_completed = $testCounts.completed
    test_passes = $testCounts.passes
    test_failures = $testCounts.failures
    test_frames = $testCounts.frames
    train_verification = $trainVerificationResult
    val_verification = $valVerificationResult
    test_verification = $testVerificationResult
    free_gb = $freeGb
    restart_attempted = $false
}

$restartable = $stageState -notin @("PASS", "FAILED", "LOW_SUCCESS_RATE")
$readyForTest = (
    $trainVerificationResult -eq "PASS" -and
    $valVerificationResult -eq "PASS" -and
    $testVerificationResult -ne "PASS"
)
if (-not $running -and $readyForTest -and -not $NoRestart) {
    $arguments = "`"$ProjectRoot\collect_v9_dataset.py`" " +
        "--output-dir `"$StageRoot\test`" --episodes 40 " +
        "--seed 20270131 --random-targets --max-iterations 1600"
    if (Test-Path -LiteralPath (Join-Path $StageRoot "test")) {
        $arguments += " --resume"
    }
    $started = Start-Process -FilePath $Python -ArgumentList $arguments `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
    $summary.restart_attempted = $true
    $summary.restart_pid = $started.Id
    $summary.restart_reason = "start test after train/val verification"
    try {
        $stage = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
        $stage.status = "running"
        $stage.splits | Add-Member -NotePropertyName test -NotePropertyValue (
            [pscustomobject]@{
                status = "running"
                episodes = 40
                seed = 20270131
                output_dir = (Join-Path $StageRoot "test")
                started_at = (Get-Date).ToString("o")
            }
        ) -Force
        $stage | ConvertTo-Json -Depth 10 |
            Set-Content -LiteralPath $StatusPath -Encoding utf8
    } catch {
        Write-MonitorLog "stage_status_update_error=$($_.Exception.Message)"
    }
    Write-MonitorLog "test_started=true pid=$($started.Id)"
    $running = $true
} elseif (-not $running -and $restartable -and -not $NoRestart) {
    Start-Sleep -Seconds 10
    $stillStopped = -not @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match "python" -and
                $_.CommandLine -match "run_stage1_collection\.py|collect_v9_dataset\.py"
            }
    ).Count
    if ($stillStopped) {
        $arguments = "`"$Runner`" --output-root `"$StageRoot`" --min-success-rate 0.9 --max-iterations 1600"
        $started = Start-Process -FilePath $Python -ArgumentList $arguments `
            -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru
        $summary.restart_attempted = $true
        $summary.restart_pid = $started.Id
        $running = $true
        Write-MonitorLog "restart_attempted=true pid=$($started.Id)"
    }
}

$summary.running_after_check = $running
$summary | ConvertTo-Json | Set-Content -LiteralPath $MonitorStatusPath -Encoding utf8
Write-MonitorLog (
    "running=$running stage=$stageState completed=$completed " +
    "passes=$passes failures=$failures frames=$frames free_gb=$freeGb " +
    "val_completed=$($valCounts.completed) val_passes=$($valCounts.passes) " +
    "val_failures=$($valCounts.failures) test_completed=$($testCounts.completed) " +
    "train_verify=$trainVerificationResult val_verify=$valVerificationResult " +
    "test_verify=$testVerificationResult " +
    "last_progress=$lastProgress"
)
