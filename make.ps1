<#
.SYNOPSIS
  Windows PowerShell mirror of the Makefile targets.
.EXAMPLE
  ./make.ps1 setup ; ./make.ps1 figures ; ./make.ps1 test
#>
param(
  [Parameter(Position = 0)]
  [ValidateSet('setup','install','figures','test','test-all','lint','format',
               'typecheck','report','submission','ci','clean','help')]
  [string]$Target = 'help'
)

$ErrorActionPreference = 'Stop'
$py = if (Test-Path './.venv/Scripts/python.exe') { './.venv/Scripts/python.exe' } else { 'python' }

switch ($Target) {
  'setup' {
    python -m venv .venv
    ./.venv/Scripts/python -m pip install -U pip
    ./.venv/Scripts/python -m pip install -e ".[dev,io,data]"
  }
  'install'   { & $py -m pip install -e ".[dev]" }
  'figures'   { & $py -m experiments.run_all }
  'test'      { & $py -m pytest -q -m "not slow" }
  'test-all'  { & $py -m pytest -q }
  'lint'      { & $py -m ruff check . }
  'format'    { & $py -m ruff format . }
  'typecheck' { & $py -m mypy src/cvlab }
  'report'    { Push-Location report; try { latexmk -pdf main.tex; latexmk -pdf supplementary.tex } finally { Pop-Location } }
  'submission' {
    & $py -m experiments.run_all
    Push-Location report; try { latexmk -pdf main.tex } finally { Pop-Location }
    & $py scripts/build_submission.py
  }
  'ci' {
    & $py -m ruff check .
    & $py -m mypy src/cvlab
    & $py -m pytest -q
  }
  'clean' {
    Get-ChildItem -Recurse -Directory -Include __pycache__,.pytest_cache,.mypy_cache,.ruff_cache,build,dist,*.egg-info |
      Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  }
  default {
    Write-Host "Targets: setup install figures test test-all lint format typecheck report submission ci clean"
  }
}
