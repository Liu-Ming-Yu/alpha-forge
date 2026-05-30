#requires -Version 5.1
<#
.SYNOPSIS
  One-command full-stack deploy: Postgres + Redis + the operator API serving
  BOTH the JSON API and the built browser console (ADR-013).

.DESCRIPTION
  Builds the Docker image (backend + frontend in one multi-stage build),
  bootstraps required secrets in .env without clobbering existing values,
  applies database migrations, brings the stack up, and prints the console URL
  and API key once the API reports healthy. No host Node or Python required.

.PARAMETER Workers
  Also start the background maintenance worker (compose profile "workers").

.PARAMETER Paper
  Also start the IBKR paper-trading engine (compose profile "paper").
  Requires TWS/Gateway reachable per the .env QP__BROKER__* settings.

.PARAMETER Rebuild
  Force a clean image rebuild (docker build --no-cache).

.EXAMPLE
  pwsh scripts/deploy.ps1
#>
[CmdletBinding()]
param(
    [switch]$Workers,
    [switch]$Paper,
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Write-Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Ok($m) { Write-Host "    $m" -ForegroundColor Green }
function Write-Note($m) { Write-Host "    $m" -ForegroundColor Yellow }

$envPath = Join-Path $repoRoot ".env"
$examplePath = Join-Path $repoRoot ".env.example"

function Get-EnvValue($key) {
    if (-not (Test-Path $envPath)) { return $null }
    $hit = Select-String -Path $envPath -Pattern ("^\s*" + [regex]::Escape($key) + "\s*=") |
        Select-Object -First 1
    if (-not $hit) { return $null }
    return (($hit.Line -split "=", 2)[1]).Trim()
}

function Set-EnvValue($key, $value) {
    if (Test-Path $envPath) { $lines = @(Get-Content $envPath) } else { $lines = @() }
    $pattern = "^\s*" + [regex]::Escape($key) + "\s*="
    $found = $false
    $out = @()
    foreach ($line in $lines) {
        if ($line -match $pattern) { $out += "$key=$value"; $found = $true } else { $out += $line }
    }
    if (-not $found) { $out += "$key=$value" }
    # UTF-8 without BOM — a BOM would corrupt the first .env key for compose.
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($envPath, $out, $utf8NoBom)
}

function New-Secret([int]$bytes = 32, [switch]$Hex) {
    $buf = New-Object byte[] $bytes
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($buf)
    if ($Hex) { return (($buf | ForEach-Object { $_.ToString("x2") }) -join "") }
    return ([Convert]::ToBase64String($buf) -replace '[+/=]', '')
}

# --- 0. Preflight ----------------------------------------------------------
Write-Step "Checking Docker"
try { docker info *> $null } catch { throw "Docker not found. Install Docker Desktop and start it." }
if ($LASTEXITCODE -ne 0) { throw "Docker daemon not reachable. Start Docker Desktop and retry." }
docker compose version *> $null
if ($LASTEXITCODE -ne 0) { throw "Docker Compose v2 ('docker compose') not found. Update Docker Desktop." }
Write-Ok "Docker is running"

# --- 1. Bootstrap .env -----------------------------------------------------
Write-Step "Preparing .env"
if (-not (Test-Path $envPath)) {
    if (Test-Path $examplePath) {
        Copy-Item $examplePath $envPath
        Write-Ok "Created .env from .env.example"
    }
    else {
        New-Item -ItemType File -Path $envPath | Out-Null
        Write-Ok "Created empty .env"
    }
}

$pg = Get-EnvValue "POSTGRES_PASSWORD"
if ([string]::IsNullOrWhiteSpace($pg) -or $pg -eq "change_me_before_running_compose") {
    $pg = New-Secret -bytes 24 -Hex
    Set-EnvValue "POSTGRES_PASSWORD" $pg
    Write-Ok "Generated POSTGRES_PASSWORD"
}
else { Write-Ok "POSTGRES_PASSWORD already set (kept)" }

$apiKey = Get-EnvValue "QP__API__OPERATOR_API_KEY"
if ([string]::IsNullOrWhiteSpace($apiKey)) {
    $apiKey = New-Secret -bytes 32
    Set-EnvValue "QP__API__OPERATOR_API_KEY" $apiKey
    Write-Ok "Generated QP__API__OPERATOR_API_KEY"
}
else { Write-Ok "QP__API__OPERATOR_API_KEY already set (kept)" }

# --- 2. Build (backend + console) -----------------------------------------
Write-Step "Building images (backend + console SPA)"
$commit = (git rev-parse --short HEAD 2>$null)
if (-not $commit) { $commit = "unknown" }
$env:QP_GIT_COMMIT = $commit
$env:QP_BUILD_DATE = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$buildArgs = @("compose", "build")
if ($Rebuild) { $buildArgs += "--no-cache" }
& docker @buildArgs
if ($LASTEXITCODE -ne 0) { throw "docker compose build failed" }
Write-Ok "Image built"

# --- 3. Datastores ---------------------------------------------------------
Write-Step "Starting Postgres and Redis"
docker compose up -d postgres redis
if ($LASTEXITCODE -ne 0) { throw "failed to start datastores" }

Write-Step "Waiting for Postgres to accept connections"
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    docker compose exec -T postgres pg_isready -U quant -d quant_platform *> $null
    if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    Start-Sleep -Seconds 2
}
if (-not $ready) { throw "Postgres did not become ready in time" }
Write-Ok "Postgres ready"

# --- 4. Migrations ---------------------------------------------------------
Write-Step "Applying database migrations"
docker compose run --rm --no-deps quant-platform-api python -m quant_platform migrate
if ($LASTEXITCODE -ne 0) { throw "database migration failed" }
Write-Ok "Schema migrated"

# --- 5. API (+ optional workers) ------------------------------------------
Write-Step "Starting the operator API"
$upArgs = @("compose")
if ($Workers) { $upArgs += @("--profile", "workers") }
if ($Paper) { $upArgs += @("--profile", "paper") }
$upArgs += @("up", "-d")
& docker @upArgs
if ($LASTEXITCODE -ne 0) { throw "failed to start services" }

# --- 6. Wait for API health ------------------------------------------------
Write-Step "Waiting for the API to report ready"
$apiReady = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:8000/health/ready" `
            -Headers @{ "X-API-Key" = $apiKey } -UseBasicParsing -TimeoutSec 3
        if ($resp.StatusCode -eq 200) { $apiReady = $true; break }
    }
    catch { }
    Start-Sleep -Seconds 2
}

# --- 7. Report -------------------------------------------------------------
Write-Host ""
if ($apiReady) {
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  Quant platform is up." -ForegroundColor Green
    Write-Host "  Console : http://localhost:8000/app/" -ForegroundColor Green
    Write-Host "  API     : http://localhost:8000/" -ForegroundColor Green
    Write-Host "  API key : $apiKey" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  Open the console and paste the API key into the connect screen."
}
else {
    Write-Note "API did not report ready within the timeout. Recent logs:"
    docker compose logs --tail 40 quant-platform-api
    throw "Build/up completed but the API is not healthy yet — see logs above."
}
