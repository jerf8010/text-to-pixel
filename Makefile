.PHONY: data train train-encdec test lint api web-test web-angular angular-build app-test

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

api:  ## API + web UI on http://localhost:8000 (loads the models in app/backend/models.yaml)
	python -m uvicorn --app-dir app/backend pixelgen_api.main:app --port 8000

web-test:  ## unit tests of the web UI logic (needs Node 20+, no npm packages)
	node --test "app/web/tests/*.test.js"

app-test: web-test
	python -m pytest -q app/backend/api_tests

# --- Optional Angular version of the UI (needs npm) ---

web-angular:  ## Angular dev server on http://localhost:4200
	cd app/frontend && npm start

angular-build:  ## build the Angular UI; serve it with PIXELGEN_STATIC_DIR=app/frontend/dist/pixelgen-app/browser
	cd app/frontend && npm install && npm run build
