.PHONY: data train test lint

data:
	python scripts/prepare_data.py --config configs/data/shapes_16.yaml

train:
	python scripts/train.py

test:
	pytest -q

lint:
	ruff check src tests scripts
