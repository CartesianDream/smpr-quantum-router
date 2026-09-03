$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONHASHSEED = "0"

python -m smpr_router run `
  --config .\configs\frozen_v5.toml `
  --executor persistent `
  --case-indices 1,25,28,53,56 `
  --workers 4 `
  --quality-reference .\evidence\reference\v5_cases.csv `
  --quality-policy exact `
  --output-dir .\results\v5_smoke
if ($LASTEXITCODE -ne 0) {
  throw "V5 smoke test failed."
}

