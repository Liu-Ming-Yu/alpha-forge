#requires -Version 5.1
<#
.SYNOPSIS
  Install the IBKR TWS API Python client (`ibapi`) into the project venv.

.DESCRIPTION
  `ibapi` is NOT published on PyPI, so `pip install -e ".[...]"` cannot provide
  it. This script downloads the official, pinned TWS API release from IBKR,
  extracts the bundled `pythonclient`, and installs it into the target Python.
  It is idempotent: if `ibapi` already imports it does nothing unless -Force.

.PARAMETER Version
  TWS API release in IBKR's zip-encoding (default 1046.01 = API 10.46.1).
  IBKR rotates hosted builds; if the default 404s, pass a current one from
  https://interactivebrokers.github.io/ (e.g. 1045.01 stable, 1047.01 latest).

.PARAMETER Python
  Python executable to install into. Default: .venv\Scripts\python.exe.

.PARAMETER Force
  Reinstall even if `ibapi` already imports.

.EXAMPLE
  pwsh scripts/install_ibapi.ps1
.EXAMPLE
  pwsh scripts/install_ibapi.ps1 -Version 1047.01 -Force
#>
[CmdletBinding()]
param(
    [string]$Version = "1046.01",
    [string]$Python,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # huge speedup for Invoke-WebRequest
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $Python) { $Python = Join-Path $repoRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path $Python)) {
    throw "Python not found at '$Python'. Create the venv first (scripts\setup.ps1) or pass -Python."
}

if (-not $Force) {
    & $Python -c "import ibapi" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $v = (& $Python -c "from ibapi import get_version_string; print(get_version_string())" 2>$null)
        Write-Host "ibapi already installed ($v). Use -Force to reinstall." -ForegroundColor Green
        exit 0
    }
}

$url = "https://interactivebrokers.github.io/downloads/twsapi_macunix.$Version.zip"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("ibapi_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    $zip = Join-Path $tmp "twsapi.zip"
    Write-Host "Downloading TWS API $Version ..." -ForegroundColor Cyan
    Write-Host "  $url"
    try {
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    }
    catch {
        throw "Download failed for version '$Version' ($_). IBKR may have removed this build; " +
        "pick a current version from https://interactivebrokers.github.io/ and pass it via -Version."
    }

    Write-Host "Extracting ..." -ForegroundColor Cyan
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $client = Join-Path $tmp "IBJts\source\pythonclient"
    if (-not (Test-Path (Join-Path $client "setup.py"))) {
        throw "pythonclient/setup.py not found in archive — the TWS API layout for $Version may differ."
    }

    Write-Host "Installing ibapi into $Python ..." -ForegroundColor Cyan
    & $Python -m pip --version *> $null
    if ($LASTEXITCODE -eq 0) {
        & $Python -m pip install $client
    }
    else {
        Write-Host "  pip not available in venv; falling back to uv ..." -ForegroundColor Yellow
        uv pip install --python $Python $client
    }
    if ($LASTEXITCODE -ne 0) { throw "ibapi install failed." }

    $ver = (& $Python -c "from ibapi import get_version_string; print(get_version_string())")
    Write-Host "ibapi $ver installed OK." -ForegroundColor Green
}
finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
