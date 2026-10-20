"""Genera el dataset sintético de formas y figuras de comprobación.

Uso:
    python scripts/prepare_data.py --config configs/data/shapes_16.yaml
"""
import argparse
from pathlib import Path

import yaml

from pixelgen.data.generate import generate_splits, scenes_to_arrays, write_dataset
from pixelgen.data.viz import save_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/shapes_16.yaml")
    parser.add_argument("--out", default="data/processed")
    parser.add_argument("--figures", default="reports/figures")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    spec, splits = generate_splits(cfg)
    out_dir = Path(args.out) / cfg["name"]
    manifest = write_dataset(cfg, spec, splits, out_dir)

    print(f"Dataset '{cfg['name']}' escrito en {out_dir}")
    for name, count in manifest["counts"].items():
        print(f"  {name:10s} {count:>8d}  sha256={manifest['sha256'][name][:12]}")
    print(f"  vocabulario de texto: {len(manifest['text_vocab'])} palabras")

    palette = spec.palette_rgb()
    for name in ("train", "test_comp"):
        arrays = scenes_to_arrays(splits[name][:32], spec)
        save_grid(
            arrays["images"],
            arrays["captions"].tolist(),
            palette,
            Path(args.figures) / f"{cfg['name']}_{name}.png",
            title=f"{cfg['name']} / {name}",
        )
    print(f"Figuras de comprobación en {args.figures}/")


if __name__ == "__main__":
    main()
