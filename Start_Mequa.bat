@echo off
rem Fallback launcher that needs no Rust: starts the dashboard and opens the browser.
cd /d "%~dp0"
start "" http://localhost:8501
python -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
