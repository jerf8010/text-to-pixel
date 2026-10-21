"""Evaluate a trained run (either architecture) and log the validity metrics to MLflow.

Usage (from the project root):
    python scripts/evaluate.py --run-id <run_id> [--n-samples 500]
"""
import argparse
import json
import math
from pathlib import Path

import mlflow
import numpy as np
import torch

from pixelgen.data.generate import build_spec
from pixelgen.data.tokenizer import TextTokenizer
from pixelgen.evaluation.generators import (
    decoder_only_generator,
    encoder_decoder_generator,
)
from pixelgen.evaluation.validity import evaluate_validity
from pixelgen.models.factory import build_model
from pixelgen.training.loop import get_device

TEST_SPLITS = ("test_iid", "test_comp")
GENERATORS = {
    "decoder_only": decoder_only_generator,
    "encoder_decoder": encoder_decoder_generator,
}
SIZE_KEYS = ("text_vocab_size", "n_colors", "max_len")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    run = mlflow.get_run(args.run_id)
    data_dir = Path(run.data.params["data_dir"])
    device = get_device()

    # Rebuild the trained model from the checkpoint stored in the run.
    ckpt_path = mlflow.artifacts.download_artifacts(
        run_id=args.run_id, artifact_path="checkpoint.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = dict(ckpt["model_config"])
    sizes = {k: mcfg.pop(k) for k in SIZE_KEYS}
    kind = mcfg["type"]
    model = build_model(mcfg, **sizes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Everything about the data is derived from the manifest.
    manifest = json.loads((data_dir / "manifest.json").read_text())
    tok = TextTokenizer(manifest["text_vocab"])
    spec = build_spec(manifest["config"])
    assert sizes["text_vocab_size"] == tok.vocab_size, "vocabulary mismatch"
    assert sizes["n_colors"] == len(manifest["palette_rgb"]), "palette mismatch"

    generate_fn = GENERATORS[kind](
        model, tok, max_len=sizes["max_len"], n_colors=sizes["n_colors"],
        device=device, temperature=args.temperature,
    )

    metrics = {}
    for split in TEST_SPLITS:
        arrays = dict(np.load(data_dir / f"{split}.npz"))
        result = evaluate_validity(
            arrays, spec, generate_fn, n_samples=args.n_samples, seed=args.seed
        )
        print(f"{split}: " + "  ".join(f"{k}={v:.3f}" for k, v in result.items()))
        for key, value in result.items():
            if key != "n_samples" and not math.isnan(value):
                metrics[f"{key}_{split}"] = value

    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "eval_n_samples": args.n_samples,
                "eval_temperature": args.temperature,
                "eval_seed": args.seed,
            }
        )


if __name__ == "__main__":
    main()