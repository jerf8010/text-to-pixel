.PHONY: data train train-encdec test lint api web app-build app-test

data:
	python scripts/prepare_data.py --config configs/data/shapes_16.yaml

train:
	python scripts/train.py --config configs/experiments/decoder_only_base.yaml

train-encdec:
	python scripts/train.py --config configs/experiments/encoder_decoder_base.yaml

test:
	python -m pytest -q

lint:
	ruff check src tests scripts app/backend

# --- Web app (see app/README.md) ---

api:  ## start the API; loads the models listed in app/backend/models.yaml
	python -m uvicorn --app-dir app/backend pixelgen_api.main:app --port 8000

web:  ## start the Angular dev server on http://localhost:4200
	cd app/frontend && npm start

app-build:  ## build the frontend so the API can serve it at http://localhost:8000
	cd app/frontend && npm install && npm run build

app-test:
	python -m pytest -q app/backend/api_tests
	cd app/frontend && npm test -- --no-watch --browsers=ChromeHeadless
