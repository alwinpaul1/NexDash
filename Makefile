# NexDash — developer task runner.
#
# Usage:
#   make setup   create a virtualenv and install dependencies
#   make data    generate the synthetic dataset
#   make train   run the end-to-end training/evaluation pipeline
#   make test    run the test suite
#   make serve   launch the FastAPI API server
#   make agent   start the interactive dispatcher REPL

VENV ?= .venv
PYTHON ?= python3
BIN := $(VENV)/bin

.PHONY: setup data train test serve agent clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .

data:
	$(BIN)/python -m nexdash.data_gen --n 6000 --out data/dataset.csv

train:
	$(BIN)/python run_pipeline.py

test:
	$(BIN)/pytest -q

serve:
	$(BIN)/python dashboard/server.py

agent:
	$(BIN)/python -m nexdash.cli

clean:
	rm -rf $(VENV) *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
