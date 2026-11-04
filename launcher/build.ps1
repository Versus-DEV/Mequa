$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
cargo build --release
$exe = Join-Path $PSScriptRoot "target\release\mequa_launcher.exe"
Copy-Item $exe (Join-Path $PSScriptRoot "..\Mequa.exe") -Force
Write-Host "Built ..\Mequa.exe  (double-click it from the Mequa folder)"
