# Datos

Esta carpeta no se versiona (ver `.gitignore`). Para regenerarla:

```bash
python scripts/prepare_data.py --config configs/data/shapes_16.yaml
```

Salida en `processed/shapes_v1/`:

- `train.npz`, `val.npz`, `test_iid.npz`, `test_comp.npz`
- `manifest.json` (configuración, semilla, conteos, hash SHA-256 de cada split, vocabulario)

Cada `.npz` contiene:

| Clave | Forma | Descripción |
|---|---|---|
| `images` | (N, 16, 16) uint8 | Índices de paleta: 0 = fondo, 1..8 = colores |
| `captions` | (N,) str | Texto, función pura de los atributos |
| `n_obj` | (N,) | 1 o 2 objetos |
| `color`, `shape`, `size` | (N, 2) | Índice por objeto (-1 si no existe) |
| `pos` | (N,) | Posición (1 objeto), -1 si no aplica |
| `rel` | (N,) | Relación (2 objetos), -1 si no aplica |
| `jitter` | (N, 2, 2) | Desplazamiento (dy, dx) de cada objeto |

El orden de `train.npz` está barajado con la semilla: los prefijos (1k, 10k, 100k)
son conjuntos anidados para el experimento de tamaño de datos.

Fuentes y licencias: ver `../DATA_LICENSES.md`.
