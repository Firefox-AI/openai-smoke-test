VENV=.venv

.PHONY: all setup install lint smoke clean

all: setup

setup:
	uv venv
	uv sync --all-groups
	uv run pre-commit install
	@echo ""
	@echo "✅ Setup complete! To activate your environment, run:"
	@echo "   source $(VENV)/bin/activate"

install:
	uv pip install --no-cache-dir -e .

lint:
	uv run ruff check .

smoke:
	$(VENV)/bin/openai-smoketest

clean:
	rm -rf __pycache__ .cache $(VENV)
