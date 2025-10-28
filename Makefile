SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

.DEFAULT_GOAL := help

MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules

HELP_WIDTH ?= 26

PYTHON ?= python3
PACKAGE ?= leadlag
SRC ?= src
TESTS ?= tests
RESULTS_ROOT ?= results
VENV ?= .venv
SMOKE_SCENARIO ?= fixed_30
SMOKE_ROOT ?= tmp_cli_json_run/smoke
SCENARIO ?= configs/scenarios/rl_ppo.yaml
RUN_ARGS ?=
TRAIN_ARGS ?=
EVAL_ARGS ?=

ifeq ($(OS),Windows_NT)
BIN_DIR := $(VENV)/Scripts
else
BIN_DIR := $(VENV)/bin
endif

VENV_PY := $(BIN_DIR)/python
VENV_PIP := $(BIN_DIR)/pip
PYTHON_RUN = $(if $(wildcard $(VENV_PY)),$(VENV_PY),$(PYTHON))
PIP_RUN = $(PYTHON_RUN) -m pip
PYTEST = $(PYTHON_RUN) -m pytest

REQS ?= requirements.txt
DEV_REQS ?= requirements-dev.txt
VENV_STAMP := $(VENV)/.venv-stamp

-include .env

export LEADLAG_RESULTS_ROOT ?= $(RESULTS_ROOT)
export RESULTS_ROOT

.PHONY: help venv install dev-install sync format lint typecheck type test coverage precommit smoke run train eval all clean distclean

help: ## Show available targets
	@printf "Targets:\n"
	@grep -E '^[[:alnum:]_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk -F ':|##' '{printf "  \\033[36m%-$(HELP_WIDTH)s\\033[0m %s\n", $$1, $$NF}'

$(VENV_STAMP):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip setuptools wheel
	touch $(VENV_STAMP)

venv: $(VENV_STAMP) ## Create the virtual environment

install: venv ## Install project dependencies
	@if [ -f $(REQS) ]; then $(VENV_PIP) install -r $(REQS); fi
	$(VENV_PIP) install -e .

dev-install: install ## Install project + development tooling
	@if [ -f $(DEV_REQS) ]; then $(VENV_PIP) install -r $(DEV_REQS); fi

sync: dev-install ## Install optional extras (RL/Kaggle)
	@if [ -f requirements-rl.txt ]; then $(VENV_PIP) install -r requirements-rl.txt; fi
	@if [ -f requirements-kaggle.txt ]; then $(VENV_PIP) install -r requirements-kaggle.txt; fi

format: ## Format code (Ruff formatter)
	$(PYTHON_RUN) -m ruff format .

lint: ## Run Ruff lint checks
	$(PYTHON_RUN) -m ruff check .

typecheck: ## Type-check code with mypy
	$(PYTHON_RUN) -m mypy $(SRC) $(TESTS)

type: typecheck ## Alias for typecheck

test: ## Run the test suite
	$(PYTEST) -q

coverage: ## Run tests with coverage reporting
	$(PYTEST) --cov=$(PACKAGE) --cov-report=term-missing

precommit: ## Run pre-commit on all files
	$(PYTHON_RUN) -m pre_commit run --all-files

smoke: RESULTS_ROOT := $(SMOKE_ROOT)
smoke: ## Run a quick smoke test scenario
	$(PYTHON_RUN) -m leadlag.main --results-root $(RESULTS_ROOT) --include $(SMOKE_SCENARIO) --max-scenarios 1 --stop-on-error $(RUN_ARGS)

run: ## Run leadlag driver (set RUN_ARGS='--include fixed_30')
	$(PYTHON_RUN) -m leadlag.main --results-root $(RESULTS_ROOT) $(RUN_ARGS)

train: ## Train RL policy (requires requirements-rl extras)
	$(PYTHON_RUN) -m leadlag.training.run_rl --config $(SCENARIO) --out $(RESULTS_ROOT) $(TRAIN_ARGS)

eval: ## Compare aggregate scenario metrics
	$(PYTHON_RUN) -m leadlag.reporting.compare_scenarios --results_root $(RESULTS_ROOT) $(EVAL_ARGS)

all: format lint typecheck test ## Run format, lint, typecheck, and tests

clean: ## Remove caches and smoke artifacts
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.py[co]' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache $(SMOKE_ROOT)

distclean: clean ## Remove virtualenv and build artifacts
	rm -rf $(VENV) $(VENV_STAMP) build dist *.egg-info tmp_cli_json_run $(RESULTS_ROOT)

