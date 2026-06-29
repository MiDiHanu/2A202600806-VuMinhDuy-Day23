.PHONY: install test lint typecheck run-scenarios grade-local diagram demo-resume visualize clean

install:
	pip install -e '.[dev,google,sqlite]'

test:
	pytest

lint:
	ruff check src tests

typecheck:
	mypy src

run-scenarios:
	python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json

grade-local:
	python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json

diagram:
	python -m langgraph_agent_lab.cli export-diagram --output reports/graph.mermaid

demo-resume:
	python -m langgraph_agent_lab.cli demo-resume

visualize:
	python -m langgraph_agent_lab.cli visualize --trace outputs/trace.json --output outputs/visualization.html

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info outputs/*.json
