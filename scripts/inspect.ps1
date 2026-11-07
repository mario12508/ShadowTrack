param([switch]$Strict)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv/Scripts/python.exe'
foreach ($tool in @('ffmpeg', 'ffprobe')) {
    $localExe = Join-Path $projectRoot ".tools/$tool.exe"
    if (Test-Path -LiteralPath $localExe) {
        [Environment]::SetEnvironmentVariable("SHADOWTRACK_$($tool.ToUpper())", $localExe, 'Process')
    }
}
$inspectArgs = @('-m', 'shadowtrack', 'inspect', '--config', (Join-Path $projectRoot 'configs/inspect.toml'))
if ($Strict) { $inspectArgs += '--strict' }
& $pythonExe @inspectArgs
exit $LASTEXITCODE
