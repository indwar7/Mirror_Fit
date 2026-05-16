# Activate the liveportrait conda env and run the server.
# Run setup.ps1 first if you haven't.
$ErrorActionPreference = "Stop"
conda activate liveportrait
Set-Location $PSScriptRoot
python server.py
