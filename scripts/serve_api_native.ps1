#requires -Version 5.1
<#
.SYNOPSIS
  Serve the operator API NATIVELY (with `ibapi` for live IBKR broker sync)
  against the Dockerized Postgres + Redis.

.DESCRIPTION
  The Docker image has no `ibapi`, so the containerized API cannot pull live
  IBKR positions/NAV. This runs the same operator API in your venv (where
  `ibapi` is installed) on 127.0.0.1 — which the TWS API trusts by default,
  unlike a container's bridge address — while reusing the durable Postgres and
  Redis that Docker provides.

  Steps: ensure ibapi is present, start Postgres + Redis, stop the Dockerized
  API (to free the port), then `serve-api` natively.

.PARAMETER Port
  Port to serve on (default 8000).

.PARAMETER ApiHost
  Bind address (default 127.0.0.1).

.EXAMPLE
  pwsh scripts/serve_api_native.ps1
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$ApiHost = "127.0.0.1"
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$py = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No .venv found. Run scripts\setup.ps1 first." }

# 1. ensure ibapi is present (broker sync needs it)
& $py -c "import ibapi" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "ibapi not installed; installing now ..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "install_ibapi.ps1") -Python $py
}

# 2. ensure datastores are up
Write-Host "Ensuring Postgres + Redis are up ..." -ForegroundColor Cyan
docker compose up -d postgres redis | Out-Null
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    docker compose exec -T postgres pg_isready -U quant -d quant_platform *> $null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $ready) { throw "Postgres did not become ready in 60s." }

# 3. free the port — stop the Dockerized API if it is holding it
$running = docker compose ps --services --filter "status=running" 2>$null
if ($running -match "quant-platform-api") {
    Write-Host "Stopping Dockerized API to free port $Port ..." -ForegroundColor Yellow
    docker compose stop quant-platform-api | Out-Null
}

# 3b. Propagate the IBKR contracts file from .env into the PROCESS env. The
# broker-sync reads QP__LIVE_IBKR__CONTRACTS_FILE from os.environ directly (not
# pydantic settings), so .env alone is invisible to it natively — without this
# the console shows 0 positions even when the account holds some.
$envContracts = Select-String -Path "$repoRoot\.env" `
    -Pattern '^\s*QP__LIVE_IBKR__CONTRACTS_FILE\s*=' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($envContracts) {
    $val = (($envContracts.Line -split '=', 2)[1]).Trim().Trim('"')
    if ($val) {
        $resolved = if ([System.IO.Path]::IsPathRooted($val)) { $val } else { Join-Path $repoRoot $val }
        $env:QP__LIVE_IBKR__CONTRACTS_FILE = $resolved
        Write-Host "Contracts file (position mapping) -> $resolved" -ForegroundColor Green
    }
}

# 4. serve natively
Write-Host ""
Write-Host "Starting native operator API at http://${ApiHost}:$Port  (ibapi enabled)" -ForegroundColor Green
Write-Host "Console: http://${ApiHost}:$Port/app/" -ForegroundColor Green
Write-Host "Reminder: TWS / IB Gateway must be running and reachable per .env QP__BROKER__* (paper TWS = 7497)," -ForegroundColor Yellow
Write-Host "          with 127.0.0.1 in the API 'Trusted IPs'. Ctrl+C to stop." -ForegroundColor Yellow
Write-Host ""
& $py -m quant_platform serve-api --host $ApiHost --port $Port
