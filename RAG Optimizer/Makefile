.PHONY: infra-up infra-down seed test lint run benchmark

DATASET     ?= nfcorpus
MAX_QUERIES ?= 50

infra-up:
	docker compose up -d qdrant elasticsearch postgres redis

infra-down:
	docker compose down

seed:
	python scripts/seed_corpus.py

test:
	pytest tests/ -q --tb=short

run:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

benchmark:
	python benchmarks/run_benchmark.py --dataset $(DATASET) --max-queries $(MAX_QUERIES)

benchmark-ci:
	python benchmarks/run_benchmark.py --dataset $(DATASET) --max-queries $(MAX_QUERIES) --ci --baseline-file benchmarks/baselines/$(DATASET).json

lint:
	ruff check . --fix
