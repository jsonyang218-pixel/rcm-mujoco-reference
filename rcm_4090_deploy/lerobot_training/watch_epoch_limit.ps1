param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingPid,

    [Parameter(Mandatory = $true)]
    [string]$LogPath,

    [string]$StatusPath,

    [int]$MaxEpochs = 10,

    [int]$PollSeconds = 10
)

$ErrorActionPreference = 'Stop'
$LogPath = [System.IO.Path]::GetFullPath($LogPath)
if (-not $StatusPath) {
    $StatusPath = Join-Path (Split-Path -Parent $LogPath) 'training_limit_watchdog.json'
}
$StatusPath = [System.IO.Path]::GetFullPath($StatusPath)
$startedAt = Get-Date

function Write-WatchdogStatus {
    param(
        [string]$State,
        [string]$Reason
    )

    $payload = [ordered]@{
        state = $State
        reason = $Reason
        training_pid = $TrainingPid
        max_epochs = $MaxEpochs
        watchdog_started_at = $startedAt.ToString('o')
        updated_at = (Get-Date).ToString('o')
        log_path = $LogPath
    }
    $temporary = "$StatusPath.tmp"
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $StatusPath -Force
}

$process = Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue
if (-not $process) {
    Write-WatchdogStatus -State 'process_already_exited' -Reason 'target process not found'
    exit 0
}
$expectedStart = $process.StartTime
Write-WatchdogStatus -State 'watching' -Reason "waiting for epoch $MaxEpochs"

while ($true) {
    $process = Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue
    if (-not $process) {
        Write-WatchdogStatus -State 'process_exited' -Reason 'training process exited before watchdog stop'
        exit 0
    }
    if ($process.StartTime -ne $expectedStart) {
        Write-WatchdogStatus -State 'pid_reused' -Reason 'training PID was reused; refusing to stop'
        exit 1
    }

    if (Test-Path -LiteralPath $LogPath) {
        $tail = Get-Content -LiteralPath $LogPath -Tail 500 -ErrorAction SilentlyContinue
        if ($tail | Select-String -SimpleMatch "Epoch ${MaxEpochs}:") {
            Stop-Process -Id $TrainingPid -Force
            Write-WatchdogStatus -State 'stopped_at_limit' -Reason "epoch $MaxEpochs started after $MaxEpochs completed epochs"
            exit 0
        }
    }
    Start-Sleep -Seconds $PollSeconds
}
