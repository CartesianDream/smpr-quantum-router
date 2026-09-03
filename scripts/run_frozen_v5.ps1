$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONHASHSEED = "0"

python -m smpr_router run `
  --config .\configs\frozen_v5.toml `
  --executor persistent `
  --workers 4 `
  --schedule lpt `
  --quality-reference .\evidence\reference\v5_cases.csv `
  --quality-policy exact `
  --output-dir .\results\v5
if ($LASTEXITCODE -ne 0) {
  throw "V5 execution failed."
}

