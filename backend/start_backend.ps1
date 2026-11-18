$ErrorActionPreference = 'Stop'
$backend = $PSScriptRoot
$repo = Split-Path -Parent $backend
$venvs = @((Join-Path $repo '.venv\Scripts\python.exe'), (Join-Path $backend '.venv\Scripts\python.exe'))
$venvPython = $venvs | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($venvPython) {
    $python = $venvPython
    $pythonArgs = @()
} else {
    $python = 'py'
    $pythonArgs = @('-3.11')
}

& $python @pythonArgs -c 'import fastapi, uvicorn, numpy, scipy, xgboost'
if ($LASTEXITCODE -ne 0) {
    throw 'Backend dependencies are missing. Run: py -3.11 -m pip install -r backend\requirements.txt'
}

if (-not $env:PATHSNAP_EVALUATOR_DATA) {
    $localZip = Join-Path $repo 'evaluator_test.zip'
    $downloadZip = Join-Path $HOME 'Downloads\evaluator_test.zip'
    if (Test-Path -LiteralPath $localZip) { $env:PATHSNAP_EVALUATOR_DATA = $localZip }
    elseif (Test-Path -LiteralPath $downloadZip) { $env:PATHSNAP_EVALUATOR_DATA = $downloadZip }
    else { throw 'Set PATHSNAP_EVALUATOR_DATA to evaluator_test.zip or its extracted directory.' }
}

$wifi = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.InterfaceAlias -match 'Wi-Fi|WLAN|Wireless' -and $_.AddressState -eq 'Preferred' -and $_.IPAddress -notlike '169.254.*' } |
    Select-Object -First 1 -ExpandProperty IPAddress
Write-Host "Evaluator data: $env:PATHSNAP_EVALUATOR_DATA"
Write-Host 'Local backend: http://localhost:8000'
if ($wifi) { Write-Host "Laptop 2 backend URL: http://${wifi}:8000" }
else { Write-Host 'Wi-Fi IPv4 not found. Check Get-NetIPAddress after joining the hotspot.' }
Write-Host 'API logs will remain visible here. Press Ctrl+C to stop.'

Push-Location $backend
try { & $python @pythonArgs -m uvicorn app.main:app --host 0.0.0.0 --port 8000 }
finally { Pop-Location }
