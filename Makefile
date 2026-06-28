.PHONY: install demo train test lint typecheck trace

install:
	pip install -e ".[dev]"

demo:        ## run the offline end-to-end remediation loop
	agent-rx demo

train:       ## pretrain the prioritizer and print held-out metrics
	agent-rx train

trace:       ## emit agent-triage-compatible NDJSON traces
	agent-rx trace --output runs/baseline.ndjson

test:
	pytest --cov=agent_rx --cov-report=term-missing

lint:
	ruff check src tests

typecheck:
	mypy src/agent_rx
