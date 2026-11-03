.DEFAULT_GOAL := help
.PHONY: help install hooks lint typecheck unit integration check audit notebook

help: ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install: ## Frozen dev install
	uv sync --frozen --all-extras

hooks: ## Install the pre-push hook
	git config core.hooksPath .githooks

lint: ## ruff check + format check
	uv run ruff check . && uv run ruff format --check .

typecheck: ## mypy
	uv run mypy

unit: ## Fast tests, no infra
	uv run pytest tests/unit -q

integration: ## CPU integration tests
	uv run pytest tests/integration -m "integration and not slow" -q

notebook: ## Rebuild the showcase notebook and run it; outputs in notebooks/out/
	uv run python notebooks/make_notebook.py
	uv run python notebooks/execute.py

check: lint typecheck unit ## What pre-push runs

audit: ## Dependency vulnerability scan
	uv run pip-audit
