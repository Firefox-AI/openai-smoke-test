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

# usage example: make stress-mlpa-fxa host="http://0.0.0.0:8080" run-time=1m
stress-mlpa-fxa:
	$(VENV)/bin/locust -f src/stress/mlpa/fxa/mlpa.py --host=$(host) --run-time=$(run-time)

# usage example: make generate-fxa-users n-users=5 env=prod
generate-fxa-users:
	$(VENV)/bin/python src/stress/mlpa/fxa/generate_test_fxa_users.py create-tokens --n-users $(n-users) --env $(env)

# usage example: make refresh-fxa-users
# usage example: make refresh-fxa-users filename=users_prod.json
refresh-fxa-users:
	@if [ -z "$(filename)" ]; then \
		$(VENV)/bin/python src/stress/mlpa/fxa/generate_test_fxa_users.py refresh-tokens; \
	else \
		$(VENV)/bin/python src/stress/mlpa/fxa/generate_test_fxa_users.py refresh-tokens --filename $(filename); \
	fi

# usage example: make delete-fxa-users
delete-fxa-users:
	$(VENV)/bin/python src/stress/mlpa/fxa/generate_test_fxa_users.py delete-users

# usage example: make stress-mlpa-appattest host="http://0.0.0.0:8080" run-time=1m
stress-mlpa-appattest:
	$(VENV)/bin/locust -f src/stress/mlpa/appattest/mlpa.py --host=$(host) --run-time=$(run-time)

# usage example: make generate-appattest-users n-users=5
generate-appattest-users:
	$(VENV)/bin/python src/stress/mlpa/appattest/generate_test_appattest_users.py --n-users $(n-users)

clean:
	rm -rf __pycache__ .cache $(VENV)
