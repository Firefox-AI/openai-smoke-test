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

# usage example make stress-mlpa host="http://0.0.0.0:8080"
stress-mlpa:
	$(VENV)/bin/locust -f src/stress/locusfiles/mlpa.py --host=$(host)

# usage example: make generate-fxa-users n-users=5 env=prod
generate-fxa-users:
	$(VENV)/bin/python src/stress/generate_test_fxa_users.py create-tokens --n-users $(n-users) --env $(env)

# usage example: make refresh-fxa-users
refresh-fxa-users:
	$(VENV)/bin/python src/stress/generate_test_fxa_users.py refresh-tokens

clean:
	rm -rf __pycache__ .cache $(VENV)
