param([ValidateSet('preflight','tests','freeze','development','readiness','fit','evaluation','predict','unseal','analyse','report','verify','audit','status','all')][string]$Command='status')
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$project=Split-Path (Split-Path $root -Parent) -Parent
$env:PYTHONUTF8='1'
$env:PYTHONHASHSEED='0'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONPYCACHEPREFIX=Join-Path $project 'layer_allocation\cache\pycache'
$env:TEMP=Join-Path $project 'layer_allocation\tmp'
$env:TMP=$env:TEMP
$python=Join-Path $project 'Feasibility\.venv\Scripts\python.exe'
Push-Location $root
try {
    if ($Command -eq 'tests') { & $python -m pytest tests -v --basetemp=../tmp/fresh_tests --junitxml=../runs/20261121_fresh01/tests.xml }
    else { & $python run.py $Command }
    if ($LASTEXITCODE -ne 0) { throw "Fresh-validation stage failed: $Command. Preserve logs and diagnose." }
} finally { Pop-Location }
