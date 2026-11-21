param(
    [string]$PythonPath,
    [string]$TectonicPath = 'tectonic',
    [switch]$AllowDownloads
)
$ErrorActionPreference = 'Stop'
$paperRoot = $PSScriptRoot
if (-not $PythonPath) {
    $runtimeRoot = 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1119-907-0_26.1119.907.0_x64__3k8sg7r9htsxt'
    $PythonPath = Join-Path $runtimeRoot 'dependencies\python\python.exe'
    if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Supply -PythonPath with a compatible authoring Python. The historical scientific environment is not changed.' }
}
foreach ($folder in @('cache','tmp','qa')) { New-Item -ItemType Directory -Path (Join-Path $paperRoot $folder) -Force | Out-Null }
$settings = @('PYTHONUTF8','PYTHONPYCACHEPREFIX','TECTONIC_CACHE_DIR','FONTCONFIG_FILE','TEMP','TMP')
$previous = @{}
foreach ($name in $settings) { $previous[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
$initialLocation = Get-Location
try {
    $env:PYTHONUTF8 = '1'
    $env:PYTHONPYCACHEPREFIX = Join-Path $paperRoot 'cache\pycache'
    $env:TECTONIC_CACHE_DIR = Join-Path $paperRoot 'cache\tectonic'
    $env:TEMP = Join-Path $paperRoot 'tmp'
    $env:TMP = $env:TEMP
    $fontCache = (Join-Path $paperRoot 'cache\fontconfig').Replace('\','/')
    $fontSettings = '<?xml version="1.0"?><fontconfig><dir>C:/Windows/Fonts</dir><cachedir>' + [System.Security.SecurityElement]::Escape($fontCache) + '</cachedir></fontconfig>'
    $env:FONTCONFIG_FILE = Join-Path $paperRoot 'cache\fonts.conf'
    [IO.File]::WriteAllText($env:FONTCONFIG_FILE,$fontSettings)
    Set-Location -LiteralPath $paperRoot
    & $PythonPath '.\prepare_evidence.py'
    if ($LASTEXITCODE -ne 0) { throw 'Read-only evidence preparation failed.' }
    $compileArgs = @('-X','compile','--untrusted','--keep-logs','--outdir','tmp')
    if (-not $AllowDownloads) { $compileArgs += '--only-cached' }
    $compileArgs += 'GradientGuard_IEEE_Access_v1_2.tex'
    & $TectonicPath @compileArgs
    if ($LASTEXITCODE -ne 0) { throw 'Typesetting failed; preserve the log and diagnose it.' }
    & $PythonPath '.\validate_manuscript.py'
    if ($LASTEXITCODE -ne 0) { throw 'Manuscript validation failed.' }
} finally {
    Set-Location $initialLocation
    foreach ($name in $settings) { [Environment]::SetEnvironmentVariable($name,$previous[$name],'Process') }
}
