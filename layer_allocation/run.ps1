param([ValidateSet('preflight','freeze','extract','features','reproduce','fit','predict','analyse','report','verify','status','all','tests','audit')][string]$Command='status')
$ErrorActionPreference='Stop'
$studyRoot=$PSScriptRoot
$env:PYTHONUTF8='1'
$env:PYTHONHASHSEED='0'
$env:PYTHONPYCACHEPREFIX=Join-Path $studyRoot 'cache\pycache'
$env:TEMP=Join-Path $studyRoot 'tmp'
$env:TMP=$env:TEMP
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$python=Join-Path (Split-Path $studyRoot -Parent) 'Feasibility\.venv\Scripts\python.exe'
Push-Location $studyRoot
try {
    if ($Command -eq 'tests') { & $python -m pytest tests -v --junitxml=runs/20261121_structural01/tests.xml }
    elseif ($Command -eq 'audit') { & $python audit_results.py }
    else { & $python run.py $Command }
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Command. Preserve logs and diagnose before proceeding." }
    if ($Command -eq 'all') {
        & $python audit_results.py
        if ($LASTEXITCODE -ne 0) { throw 'Completed-output audit failed; preserve this attempt and diagnose.' }
    }
} finally { Pop-Location }
