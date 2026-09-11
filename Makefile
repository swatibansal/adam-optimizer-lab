# Optimizer Lab — build/run entry points.
# All targets use the project virtualenv in .venv so nothing is installed globally.

PY := .venv/bin/python

.PHONY: setup dev test lint exp1 exp2 exp3 exp4 exp5 all report clean

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e .

dev: setup
	.venv/bin/python -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src experiments tests
	$(PY) -m ruff format --check src experiments tests

exp1:
	$(PY) experiments/exp1_adam_by_hand.py

exp2:
	$(PY) experiments/exp2_bias_correction.py

exp3:
	$(PY) experiments/exp3_update_ratio.py

exp4:
	$(PY) experiments/exp4_cosine_vs_wsd.py

exp5:
	$(PY) experiments/exp5_width_sweep.py

all: exp1 exp2 exp3 exp4 exp5 report

report:
	$(PY) experiments/report.py

clean:
	rm -rf runs/*
