$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python -m pip install --upgrade maturin
python -m maturin build --release
$whl = Get-ChildItem "target\wheels\*.whl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
python -m pip install --force-reinstall $whl.FullName
python -c "import mequa_rs; print('mequa_rs ok, rayon threads:', mequa_rs.num_threads())"
