$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ScriptRoot ".venv\Scripts\python.exe"
$StatusPath = Join-Path $ScriptRoot "results\parametric_many_query_flat_stiffened_status.json"

function Write-Status {
    param(
        [string]$Case,
        [string]$Stage,
        [string]$State,
        [string]$OutputRoot,
        [string]$Message = ""
    )
    $payload = [ordered]@{
        updated_at = (Get-Date).ToString("s")
        case = $Case
        stage = $Stage
        state = $State
        output_root = $OutputRoot
        message = $Message
    }
    $payload | ConvertTo-Json | Set-Content -Path $StatusPath -Encoding UTF8
}

function Invoke-LoggedNative {
    param(
        [string]$LogPath,
        [string[]]$Arguments
    )
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $Python @Arguments >> $LogPath 2>&1
        $nativeExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($nativeExitCode -ne 0) {
        throw "Command failed with exit code $nativeExitCode`: $Python $($Arguments -join ' ')"
    }
}

Set-Location $ScriptRoot
New-Item -ItemType Directory -Force -Path (Join-Path $ScriptRoot "results") | Out-Null

$cases = @("flat", "stiffened")
foreach ($caseName in $cases) {
    $outRoot = Join-Path $ScriptRoot ("results\parametric_many_query_" + $caseName + "_5000_best")
    $caseDir = Join-Path $outRoot $caseName
    $logPath = Join-Path $outRoot "pipeline.log"
    New-Item -ItemType Directory -Force -Path $outRoot | Out-Null

    try {
        Write-Status -Case $caseName -Stage "training" -State "running" -OutputRoot $outRoot
        $trainingArguments = @(
            (Join-Path $ScriptRoot "run_parametric_many_query.py"),
            "--case", $caseName,
            "--representation", "pod",
            "--epochs", "5000",
            "--device", "cuda",
            "--out", $outRoot
        )
        $checkpointPath = Join-Path $caseDir "parametric_operator.pt"
        if ($caseName -eq "flat" -and (Test-Path $checkpointPath)) {
            $trainingArguments += "--skip-training"
        }
        Invoke-LoggedNative -LogPath $logPath -Arguments $trainingArguments

        Write-Status -Case $caseName -Stage "loss_plot" -State "running" -OutputRoot $outRoot
        Invoke-LoggedNative -LogPath $logPath -Arguments @(
            (Join-Path $ScriptRoot "plot_parametric_training_loss.py"),
            "--checkpoint", (Join-Path $caseDir "parametric_operator.pt"),
            "--out", (Join-Path $caseDir "training_loss_curve.png")
        )

        Write-Status -Case $caseName -Stage "frequency_evaluation" -State "running" -OutputRoot $outRoot
        Invoke-LoggedNative -LogPath $logPath -Arguments @(
            (Join-Path $ScriptRoot "evaluate_parametric_frequency_range.py"),
            "--case", $caseName,
            "--checkpoint", (Join-Path $caseDir "parametric_operator.pt"),
            "--frequency-min", "0",
            "--frequency-max", "200",
            "--frequency-points", "41",
            "--observation-index", "224",
            "--out", (Join-Path $caseDir "frequency_range_evaluation")
        )

        Write-Status -Case $caseName -Stage "complete" -State "done" -OutputRoot $outRoot
    }
    catch {
        Write-Status -Case $caseName -Stage "failed" -State "failed" -OutputRoot $outRoot -Message $_.Exception.Message
        throw
    }
}

Write-Status -Case "flat,stiffened" -Stage "complete" -State "done" -OutputRoot (Join-Path $ScriptRoot "results")
