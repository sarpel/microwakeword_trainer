# Makefile for microwakeword_trainer
# Run 'make help' to see available targets

.PHONY: help install install-dev lint lint-fix format format-check type-check test test-unit test-integration test-parallel test-fast coverage clean check pre-commit build check-dist

# Default target
help:
	@echo "Available targets:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  lint           Run ruff linter"
	@echo "  format         Run black formatter and ruff fixes"
	@echo "  format-check   Check formatting without making changes"
	@echo "  type-check     Run mypy type checker"
	@echo "  test           Run all tests"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  coverage       Run tests with coverage report"
	@echo "  check          Run all checks (lint, format-check, type-check, test)"
	@echo "  clean          Clean build artifacts"
	@echo "  pre-commit     Install and run pre-commit hooks"

# Installation
install:
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

# Linting (Ruff replaces flake8, pylint, isort)
lint:
	ruff check src/ config/ scripts/ tests/
	ruff check --select I src/ config/ scripts/ tests/

lint-fix:
	ruff check --fix src/ config/ scripts/ tests/
	ruff check --select I --fix src/ config/ scripts/ tests/

# Formatting (Black + Ruff)
format:
	black src/ config/ scripts/ tests/
	ruff format src/ config/ scripts/ tests/

format-check:
	black --check src/ config/ scripts/ tests/
	ruff format --check src/ config/ scripts/ tests/

# Type Checking

type-check:
	mypy src/ config/

# Testing

test:
	pytest -v

test-unit:
	pytest -v -m unit

test-integration:
	pytest -v -m integration

test-parallel:
	pytest -v -n auto

test-fast:
	pytest -v -m "not slow and not gpu" -n auto

coverage:
	pytest --cov=src --cov=config --cov-report=term-missing --cov-report=html:coverage_html

# All checks (CI pipeline)
check: lint format-check type-check test

# Pre-commit hooks
pre-commit:
	pre-commit install
	pre-commit run --all-files

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf coverage_html/
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete

# Build
build:
	python -m build

# Check package
check-dist:
	twine check dist/*
