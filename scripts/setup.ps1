#requires -Version 5.1
<#
.SYNOPSIS
  One-command setup for a new machine: Python 3.11 venv + dependencies + ibapi
  (IBKR TWS API) + .env. After this, run `scripts\serve_api_native.ps1` for the
  broker-capable API, or `scripts\deploy.ps1` for the full Docker stack.

.PARAMETER Extras
  Also install the heavy research extras: ml (XGBoost) and backtest (vectorbt).

.PARAMETER NoIbapi
  Skip installing ibapi (use if you will never connect to IBKR from this machine).

.EXAMPLE
  pwsh scripts/setup.ps1
.EXAMPLE
  pwsh scripts/setup.ps1 -Extras
#>
[CmdletBinding()]
param(
    [switch]$Extras,
    [switch]$NoIbapi
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Write-Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }

# --- 1. locate Python 3.11 -------------------------------------------------
Write-Step "Locating Python 3.11"
$pyExe = $null; $pyArgs = @()
foreach ($cand in @(@("py", "-3.11"), @("python3.11"), @("python"))) {
    $exe = $cand[0]; $a = @($cand[1..($cand.Length - 1)])
    try { $out = & $exe @a --version 2>&1 } catch { continue }
    if ($out -match "3\.11") { $pyExe = $exe; $pyArgs = $a; break }
}
if (-not $pyExe) { throw "Python 3.11 not found. Install it from https://www.python.org/downloads/ (3.11.x)." }
Write-Host "    Using: $pyExe $($pyArgs -join ' ') ($out)" -ForegroundColor Green

# --- 2. create venv --------------------------------------------------------
Write-Step "Creating .venv"
if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "    .venv already exists (kept)" -ForegroundColor Green
}
else {
    & $pyExe @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
    Write-Host "    Created .venv" -ForegroundColor Green
}
$py = Join-Path $repoRoot ".venv\Scripts\python.exe"

# --- 3. install dependencies ----------------------------------------------
Write-Step "Installing dependencies"
& $py -m pip install --upgrade pip
$spec = if ($Extras) { ".[dev,api,ml,backtest]" } else { ".[dev,api]" }
Write-Host "    pip install -e `"$spec`"" -ForegroundColor Cyan
& $py -m pip install -e $spec
if ($LASTEXITCODE -ne 0) { throw "dependency install failed." }

# --- 4. ibapi (IBKR TWS API) ----------------------------------------------
if (-not $NoIbapi) {
    Write-Step "Installing ibapi (IBKR TWS API)"
    & (Join-Path $PSScriptRoot "install_ibapi.ps1") -Python $py
}

# --- 5. .env ---------------------------------------------------------------
Write-Step "Preparing .env"
if (Test-Path ".env") {
    Write-Host "    .env already exists (kept)" -ForegroundColor Green
}
elseif (Test-Path "infra\config\settings.example.env") {
    Copy-Item "infra\config\settings.example.env" ".env"
    Write-Host "    Created .env from infra\config\settings.example.env" -ForegroundColor Green
    Write-Host "    EDIT IT: set POSTGRES_PASSWORD, QP__API__OPERATOR_API_KEY, and your API keys." -ForegroundColor Yellow
}
else {
    Write-Host "    No settings.example.env found; create .env manually." -ForegroundColor Yellow
}

# --- done ------------------------------------------------------------------
Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  Setup complete." -ForegroundColor Green
Write-Host "  Next:" -ForegroundColor Green
Write-Host "    1. Edit .env (secrets + POSTGRES_PASSWORD)."
Write-Host "    2. Broker-capable API (native):  pwsh scripts\serve_api_native.ps1"
Write-Host "       or full Docker stack:         pwsh scripts\deploy.ps1"
Write-Host "============================================================" -ForegroundColor Green
