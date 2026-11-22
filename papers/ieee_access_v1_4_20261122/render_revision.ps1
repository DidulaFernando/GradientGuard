$ErrorActionPreference = 'Stop'
$lo = Join-Path $PSScriptRoot '..\ieee_access_v1_3_20261122\cache\libreoffice'
$runtime = 'C:\Users\rangi\.cache\codex-runtimes\codex-primary-runtime\dependencies'
$renderer = 'C:\Users\rangi\.codex\plugins\cache\openai-primary-runtime\documents\26.1119.11507\skills\documents\render_docx.py'
$fontTarget = Join-Path $env:TEMP 'gradientguard-opensymbol.ttf'
Copy-Item -LiteralPath (Join-Path $lo 'Fonts\opens___.ttf') -Destination $fontTarget -Force
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class GradientGuardRenderFont { [DllImport("gdi32.dll", CharSet=CharSet.Unicode)] public static extern int AddFontResourceEx(string name, uint flags, IntPtr pdv); [DllImport("gdi32.dll", CharSet=CharSet.Unicode)] public static extern bool RemoveFontResourceEx(string name, uint flags, IntPtr pdv); }'
$added = [GradientGuardRenderFont]::AddFontResourceEx($fontTarget,0,[IntPtr]::Zero)
if ($added -lt 1) { throw 'Could not temporarily load OpenSymbol' }
try {
    $env:PATH = (Join-Path $lo 'program') + ';' + (Join-Path $runtime 'native\poppler\Library\bin') + ';' + $env:PATH
    & (Join-Path $runtime 'python\python.exe') $renderer (Join-Path $PSScriptRoot 'GradientGuard_IEEE_Access_v1_4.docx') --output_dir (Join-Path $PSScriptRoot 'qa\render') --emit_pdf
    if ($LASTEXITCODE -ne 0) { throw 'Document rendering failed' }
} finally {
    [GradientGuardRenderFont]::RemoveFontResourceEx($fontTarget,0,[IntPtr]::Zero) | Out-Null
}
