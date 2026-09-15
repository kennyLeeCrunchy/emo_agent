param(
    [string]$Python = "python",
    [string]$EnvFile = "Sites\.env.local"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Missing $EnvFile. Copy Sites/.env.example to Sites/.env.local and fill in secrets."
}

Get-Content -LiteralPath $EnvFile | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#")) { return }
    $parts = $line.Split("=", 2)
    if ($parts.Count -ne 2) { throw "Invalid environment line: $line" }
    [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1], "Process")
}

if ($env:EMO_AGENT_PUBLIC_MODE -ne "true") {
    throw "Refusing public startup because EMO_AGENT_PUBLIC_MODE is not true."
}
if (-not $env:EMO_AGENT_PROXY_SECRET -or $env:EMO_AGENT_PROXY_SECRET.Length -lt 32) {
    throw "EMO_AGENT_PROXY_SECRET must be at least 32 characters."
}

& $Python -m backend.run_fastapi --host 127.0.0.1 --port 7860
