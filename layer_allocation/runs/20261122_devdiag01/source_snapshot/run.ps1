param([ValidateSet('tests','freeze','analyse','report','verify','status','all')][string]$Command='status')
$ErrorActionPreference='Stop'
$project=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$python=Join-Path $project 'Feasibility\.venv\Scripts\python.exe'
$env:PYTHONUTF8='1'
$env:PYTHONHASHSEED='0'
$env:PYTHONPYCACHEPREFIX=Join-Path $project 'layer_allocation\cache\pycache'
$env:TEMP=Join-Path $project 'layer_allocation\tmp'
$env:TMP=$env:TEMP
Push-Location $PSScriptRoot
try {
    if ($Command -eq 'tests' -or $Command -eq 'all') {
        & $python -m pytest tests -v --basetemp=../tmp/development_diagnostic_tests --junitxml=../runs/20261122_devdiag01/tests.xml
        if ($LASTEXITCODE -ne 0) { throw 'Diagnostic tests failed; no scientific analysis launched.' }
    }
    if ($Command -eq 'all') {
        & $python diagnose.py freeze
        if ($LASTEXITCODE -ne 0) { throw 'Freeze or preservation failed.' }
        if (!(Test-Path -LiteralPath '../runs/20261122_devdiag01/analysis_complete.json')) {
            & $python diagnose.py analyse
            if ($LASTEXITCODE -ne 0) { throw 'Diagnostic analysis failed; preserve logs.' }
        }
        & $python report.py
    } elseif ($Command -eq 'report') { & $python report.py }
    elseif ($Command -ne 'tests') { & $python diagnose.py $Command }
    if ($LASTEXITCODE -ne 0) { throw "Diagnostic command failed: $Command" }
} finally { Pop-Location }
