# Activate the instantid conda env and run the server.
# Run setup.ps1 first if you haven't.
$ErrorActionPreference = "Stop"
conda activate instantid
Set-Location $PSScriptRoot
python server.py
