# cvlab -- developer entry points.  `make help` for the list.
# Windows: use `python -m` targets or Git Bash; a make.ps1 mirror is provided.

PY ?= python
PKG := cvlab

.DEFAULT_GOAL := help
.PHONY: help setup install figures test test-all lint format typecheck \
        report submission clean distclean ci

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create .venv and install with dev extras
	$(PY) -m venv .venv
	./.venv/Scripts/python -m pip install -U pip
	./.venv/Scripts/python -m pip install -e ".[dev,io,data]"

install: ## Editable install into the active environment
	$(PY) -m pip install -e ".[dev]"

figures: ## Regenerate every figure + metrics JSON (synthetic data)
	$(PY) -m experiments.run_all

test: ## Fast unit tests
	$(PY) -m pytest -q -m "not slow"

test-all: ## Every test incl. slow end-to-end smoke
	$(PY) -m pytest -q

lint: ## ruff check
	$(PY) -m ruff check .

format: ## ruff format
	$(PY) -m ruff format .

typecheck: ## mypy on the library
	$(PY) -m mypy src/$(PKG)

report: ## Build report.pdf + supplementary.pdf
	$(MAKE) -C report all

submission: figures report ## Build the graded 23i-2515_Assignment01.zip
	$(PY) scripts/build_submission.py

ci: lint typecheck test-all ## What CI runs

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ build dist *.egg-info \
	       src/*.egg-info .coverage coverage.xml
	$(MAKE) -C report clean

distclean: clean ## Also remove venv, generated figures and the zip
	rm -rf .venv output_images/*.png results/*.json *_Assignment01.zip
	$(MAKE) -C report distclean
