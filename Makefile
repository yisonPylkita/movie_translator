# Movie Translator - Development Commands

.PHONY: help lint format check clean test run-example

help: ## Show this help message
	@echo "Movie Translator - Development Commands"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint: ## Run ruff linter
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

check: ## Run both lint and format
	uv run ruff check .
	uv run ruff format .

clean: ## Clean temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "temp" -exec rm -rf {} + 2>/dev/null || true

test: ## Run quick test on example file
	@echo "Testing with example MKV file..."
	@if [ -f "~/Downloads/test_movies/SPY x FAMILY - S01E01.mkv" ]; then \
		uv run python3 translate.py "~/Downloads/test_movies/SPY x FAMILY - S01E01.mkv" --output ./test_output; \
	else \
		echo "Example file not found. Please ensure SPY x FAMILY - S01E01.mkv exists in ~/Downloads/test_movies/"; \
	fi

run-example: ## Run full pipeline on example file
	@echo "Running full pipeline on example file..."
	@if [ -f "~/Downloads/test_movies/SPY x FAMILY - S01E01.mkv" ]; then \
		uv run python3 translate.py "~/Downloads/test_movies/SPY x FAMILY - S01E01.mkv" --device mps --batch-size 16; \
	else \
		echo "Example file not found. Please ensure SPY x FAMILY - S01E01.mkv exists in ~/Downloads/test_movies/"; \
	fi

install: ## Install dependencies
	uv add pysubs2 torch transformers ruff

setup: ## Install dependencies and check setup
	uv add pysubs2 torch transformers ruff
	uv run ruff check .
