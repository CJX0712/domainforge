.PHONY: install test lint demo clean

PY = python

install:
	$(PY) -m pip install -r requirements.lock.txt

test:
	$(PY) -m pytest -q -W ignore::UserWarning -W ignore::FutureWarning

lint:
	$(PY) -m ruff format --check .
	$(PY) -m ruff check .

demo:
	$(PY) -m domainforge.examples.run_demo

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
