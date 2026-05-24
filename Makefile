.PHONY: install lint test ingest embed serve eval clean

install:
	uv sync || pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src

fmt:
	ruff format src tests
	ruff check --fix src tests

test:
	pytest -q

ingest:
	python -m asx_grounded.ingestion.fetch_asx --codes CBA,BHP,WBC,CSL,WES --days 30

embed:
	python -m asx_grounded.ingestion.embed

serve:
	uvicorn asx_grounded.api.main:app --reload --port 8000

eval:
	python -m asx_grounded.eval.run_eval

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
