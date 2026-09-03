$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:RAYON_NUM_THREADS = "1"

if (-not (Test-Path ".\v7\FROZEN_MANIFEST.json")) {
  throw "Run prepare_v7.py and freeze_v7.py before producing V7 results."
}

foreach ($repeat in 1..3) {
  $output = ".\results\v7_repeat_{0:d2}" -f $repeat
  if (Test-Path $output) {
    throw "Output already exists: $output"
  }
  python -u .\scripts\run_v7_core.py `
    --repeat $repeat `
    --output-dir $output `
    2>&1 | Tee-Object (
      ".\results\v7_repeat_{0:d2}_console.txt" -f $repeat
    )
  if ($LASTEXITCODE -ne 0) {
    throw "V7 repeat $repeat failed."
  }
}

python -u .\scripts\analyze_v7.py `
  --results-root .\results `
  --output-dir .\results\v7_analysis `
  2>&1 | Tee-Object .\results\v7_analysis_console.txt
if ($LASTEXITCODE -ne 0) {
  throw "V7 analysis failed."
}

