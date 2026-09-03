$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONHASHSEED = "0"

python -m smpr_router run `
  --config .\configs\frozen_v6.toml `
  --executor persistent `
  --workers 4 `
  --schedule lpt `
  --quality-reference .\evidence\reference\v6_cases.csv `
  --quality-policy primary `
  --output-dir .\results\v6
if ($LASTEXITCODE -ne 0) {
  throw "V6 execution failed."
}

