$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install psutil mcp tifffile
python -c "import streamlit, plotly; print('streamlit', streamlit.__version__, 'ok')"
Write-Host "Setup done. Run Start_Mequa.bat, or build the launcher: cd launcher; .\build.ps1"
