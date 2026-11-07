param(
    [ValidateSet('analyze', 'report', 'collect')][string]$Mode = 'analyze',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$SyncArgs
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
foreach ($tool in @('ffmpeg', 'ffprobe')) {
    $localExe = Join-Path $projectRoot ".tools/$tool.exe"
    if (Test-Path -LiteralPath $localExe) {
        [Environment]::SetEnvironmentVariable("SHADOWTRACK_$($tool.ToUpper())", $localExe, 'Process')
    }
}
& (Join-Path $projectRoot '.venv/Scripts/shadowtrack.exe') sync $Mode @SyncArgs
exit $LASTEXITCODE
