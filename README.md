# pixel-text2image

Generador de imágenes de pixel art a partir de texto, entrenado desde cero con un
Transformer encoder-decoder (seq2seq). Incluye un estudio de generalización
composicional (encoder-decoder vs. decoder-only) con seguimiento de experimentos en MLflow.

> Estado: en desarrollo.

## Demo
_(añadir gif o enlace al Space de Hugging Face)_

## Preguntas de investigación
- ¿Generaliza mejor a combinaciones no vistas un encoder-decoder o un decoder-only de igual tamaño?
- ¿Cómo cambia la generalización con el tamaño de los datos y la fracción de combinaciones retenidas?

## Resultados clave
_(tabla y figuras de `reports/figures/`)_

## Cómo reproducir
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make data        # genera data/processed/shapes_v1 y figuras de comprobación
make test
mlflow ui --backend-store-uri sqlite:///mlflow.db
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
python scripts/evaluate.py --run-id <run_id> --n-samples 100
```

## Datos
Dataset sintético de formas (16x16, paleta de 9 índices) con splits composicionales:
ver `configs/data/shapes_16.yaml` y `data/README.md`.

## Licencias
Código bajo la licencia indicada en `LICENSE`. Los datos tienen licencias propias:
ver `DATA_LICENSES.md`.
