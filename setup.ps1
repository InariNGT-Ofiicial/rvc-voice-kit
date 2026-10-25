#Requires -Version 5.1
<#
.SYNOPSIS
    Bootstrap the RVC voice-conversion toolchain (upstream engine + venv + deps).

.DESCRIPTION
    Creates:
        <Root>\repo   upstream RVC pinned to $Commit
        <Root>\venv   Python 3.12 virtualenv with torch cu128 + upstream deps
        <Root>\rvcx.local.toml   machine-local path config (copied from the example)

    This repository intentionally ships NO weights and NO upstream source.
    Everything large is fetched here.

.PARAMETER Root
    Working root. Must be on a drive with ~12 GB free (repo + venv + assets).

.PARAMETER Commit
    Upstream commit to pin. Default matches the version this kit was validated on.

.PARAMETER Python
    Python 3.12 interpreter used to create the venv.

.PARAMETER Cpu
    Install CPU-only torch instead of cu128 (slower, but works without NVIDIA GPU).

.PARAMETER DryRun
    Print the commands without executing them.

.EXAMPLE
    .\setup.ps1 -Root E:\RVC
#>
[CmdletBinding()]
param(
    [string]$Root = "E:\RVC",
    [string]$Commit = "81eed5e8f68b6bed1789f682fe78cdd324495afc",
    [string]$Python = "python",
    [switch]$Cpu,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Upstream = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
$Kit = $PSScriptRoot

function Step($msg) { Write-Host "`n=== $msg" -ForegroundColor Cyan }
function Run([string[]]$Cmd) {
    $line = ($Cmd -join " ")
    if ($DryRun) { Write-Host "  [dry] $line" -ForegroundColor DarkGray; return }
    Write-Host "  $line" -ForegroundColor DarkGray
    & $Cmd[0] $Cmd[1..($Cmd.Length - 1)]
    if ($LASTEXITCODE -ne 0) { throw "command failed (exit $LASTEXITCODE): $line" }
}

Write-Host "rvc-voice-kit setup" -ForegroundColor Green
Write-Host "  root    : $Root"
Write-Host "  commit  : $Commit"
Write-Host "  python  : $Python"
Write-Host "  torch   : $(if ($Cpu) { 'cpu' } else { '2.7.1+cu128' })"
if ($DryRun) { Write-Host "  MODE    : dry run, nothing will be written" -ForegroundColor Yellow }

# ---------------------------------------------------------------- 1. python
Step "1/5 check python"
if (-not $DryRun) {
    $ver = & $Python -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "python not found: $Python" }
    if ($ver -ne "3.12") {
        Write-Warning "expected Python 3.12, got $ver. RVC 2.3 upstream is validated on 3.12."
    }
    Write-Host "  python $ver OK"
} else {
    Write-Host "  [dry] $Python -c ..." -ForegroundColor DarkGray
}

# ------------------------------------------------------------------ 2. repo
Step "2/5 clone / pin upstream RVC"
$repo = Join-Path $Root "repo"
if ((Test-Path (Join-Path $repo ".git")) -and -not $DryRun) {
    Write-Host "  repo exists, fetching and checking out $Commit"
    Run @("git", "-C", $repo, "fetch", "--all", "--tags")
    Run @("git", "-C", $repo, "checkout", "--force", $Commit)
} else {
    if (-not $DryRun) { New-Item -ItemType Directory -Force -Path $Root | Out-Null }
    Run @("git", "clone", $Upstream, $repo)
    Run @("git", "-C", $repo, "checkout", "--force", $Commit)
}

# ------------------------------------------------------------------- 3. venv
Step "3/5 create virtualenv"
$venv = Join-Path $Root "venv"
$vpy = Join-Path $venv "Scripts\python.exe"
if (-not $DryRun) {
    if (-not (Test-Path $vpy)) { Run @($Python, "-m", "venv", $venv) }
    else { Write-Host "  venv exists: $venv" }
} else {
    Write-Host "  [dry] $Python -m venv $venv" -ForegroundColor DarkGray
}

# -------------------------------------------------------------- 4. packages
Step "4/5 install torch and upstream dependencies"
if (-not (Test-Path (Join-Path $repo "requirments_cu128_py312.txt")) -and -not $DryRun) {
    throw "upstream requirements file not found; checkout may have failed"
}
Run @($vpy, "-m", "pip", "install", "--upgrade", "pip")
if ($Cpu) {
    Run @($vpy, "-m", "pip", "install", "torch==2.7.1", "torchaudio==2.7.1")
} else {
    # torch must stay below 2.8 for the cu128 pair used by this kit
    Run @($vpy, "-m", "pip", "install", "torch==2.7.1+cu128", "torchaudio==2.7.1+cu128",
          "--index-url", "https://mirrors.nju.edu.cn/pytorch/whl/cu128",
          "--extra-index-url", "https://mirrors.pku.edu.cn/pypi/simple")
}
Run @($vpy, "-m", "pip", "install", "-r", (Join-Path $repo "requirments_cu128_py312.txt"))

# ---------------------------------------------------------------- 5. config
Step "5/5 write machine-local config"
$local = Join-Path $Kit "rvcx.local.toml"
$example = Join-Path $Kit "rvcx.local.example.toml"
if (-not $DryRun) {
    if (Test-Path $local) {
        Write-Host "  $local already exists, left untouched"
    } elseif (Test-Path $example) {
        Copy-Item $example $local
        Write-Host "  copied example -> $local"
        Write-Host "  EDIT IT: set root / repo / python to match $Root" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [dry] copy $example -> $local" -ForegroundColor DarkGray
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host @"

Next steps
----------
1. Edit rvcx.local.toml  (root = $Root, repo = $Root\repo, python = $venv\Scripts\python.exe)
2. Download rmvpe weights to  $repo\assets\rmvpe\rmvpe.pt
   (the WebUI fetches them on first launch, or copy from an existing install)
3. Make sure ffmpeg.exe is on PATH, or set ffmpeg_dir in rvcx.local.toml
4. Put a trained voice at  <models>\<name>\  (model.pth + added_*.index)
5. Verify:
       $vpy -m rvcx.paths
       $vpy -m rvcx.voice doctor
       $vpy -m rvcx.voice convert --voice <name> --input dry.wav --output wet.wav
"@ -ForegroundColor Gray
