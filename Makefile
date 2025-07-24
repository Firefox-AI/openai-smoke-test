PYTHON=python3
VENV=.venv
REQS=requirements.txt

.PHONY: all setup run clean qdrant-up 

all: setup

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip 
	$(VENV)/bin/pip install -r $(REQS)
	$(VENV)/bin/pip install -e .
  
smoke:
	$(VENV)/bin/openai-smoketest

clean:
	rm -rf __pycache__ .cache $(VENV)

