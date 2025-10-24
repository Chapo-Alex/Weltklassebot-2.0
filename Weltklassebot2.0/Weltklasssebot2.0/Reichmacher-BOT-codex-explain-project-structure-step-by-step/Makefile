.PHONY: fmt lint type test test-slow cov bench
fmt:
	ruff check --fix .
lint:
	ruff check .
type:
	mypy --strict
test:
	PYTHONPATH=src pytest -q -m "not slow"
test-slow:
	PYTHONPATH=src pytest -q -m slow
cov:
	PYTHONPATH=src pytest -q --cov=src --cov-report=xml
bench:
	PYTHONPATH=src python scripts/bench_backtester.py --candles 50000 --repeats 2
