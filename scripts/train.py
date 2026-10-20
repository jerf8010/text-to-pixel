"""Train the decoder-only baseline and track the run with MLflow.

Usage (from the project root):
    python scripts/train.py --config configs/experiments/decoder_only_base.yaml
"""
import argparse
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pixelgen.data.dataset import ShapesDataset
from pixelgen.data.dataset_gpt import DecoderOnlyDataset
from pixelgen.data.shapes import CANVAS
from pixelgen.data.tokenizer import TextTokenizer
from pixelgen.models.decoder_only import DecoderOnlyTransformer
from pixelgen.training.loop import evaluate, flatten_dict, get_device

SPLITS = ("train", "val", "test_iid", "test_comp")


def compute_max_len(data_dir: Path) -> int:
    """Longest caption, in words, across every split."""
    longest = 0
    for split in SPLITS:
        captions = np.load(data_dir / f"{split}.npz")["captions"]
        longest = max(longest, max(len(c.split()) for c in captions))
    return longest


def build_datasets(data_dir: Path):
    """Return the manifest and the train / val decoder-only datasets."""
    manifest = json.loads((data_dir / "manifest.json").read_text())
    tok = TextTokenizer(manifest["text_vocab"])
    max_len = compute_max_len(data_dir)
    n_colors = len(manifest["palette_rgb"])

    def make(split: str) -> DecoderOnlyDataset:
        base = ShapesDataset(data_dir / f"{split}.npz", tok, max_len, n_colors)
        return DecoderOnlyDataset(base, tok.vocab_size, max_len)

    return manifest, make("train"), make("val")


def infinite(loader):
    """Iterate over a DataLoader forever, reshuffling at every pass."""
    while True:
        yield from loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg, tcfg = cfg["model"], cfg["train"]

    torch.manual_seed(tcfg["seed"])
    device = get_device()

    manifest, train_ds, val_ds = build_datasets(Path(cfg["data_dir"]))
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False)

    # The sequence is caption (max_len) + <sep> + pixels, minus the shifted token.
    block_size = train_ds.max_len + CANVAS * CANVAS
    model = DecoderOnlyTransformer(
        vocab_size=train_ds.vocab_size, block_size=block_size, **mcfg
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("decoder-only")
    with mlflow.start_run(run_name=f"{cfg['name']}-seed{tcfg['seed']}"):
        mlflow.log_params(flatten_dict(cfg))
        mlflow.log_params(
            {
                "block_size": block_size,
                "n_params": n_params,
                "device": device,
                "dataset_sha256_train": manifest["sha256"]["train"],
            }
        )
        mlflow.log_artifact(args.config)
        print(f"device={device}  params={n_params / 1e6:.2f}M  block_size={block_size}")

        batches = infinite(train_loader)
        loss_sum, loss_count = 0.0, 0
        start = time.time()
        model.train()

        for step in range(1, tcfg["max_steps"] + 1):
            batch = next(batches)
            x, y = batch["x"].to(device), batch["y"].to(device)

            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            loss_count += 1

            if step % tcfg["eval_every"] == 0 or step == tcfg["max_steps"]:
                train_loss = loss_sum / loss_count  # mean since the last evaluation
                val_loss = evaluate(model, val_loader, device)
                loss_sum, loss_count = 0.0, 0
                elapsed = time.time() - start
                mlflow.log_metric("train_loss", train_loss, step=step)
                mlflow.log_metric("val_loss", val_loss, step=step)
                mlflow.log_metric("elapsed_s", elapsed, step=step)
                print(f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f}"
                      f" | {elapsed:.0f}s")

        checkpoint = {
            "model": model.state_dict(),
            "model_config": {**mcfg, "vocab_size": train_ds.vocab_size, "block_size": block_size},
                "data_config": {
                "max_len": train_ds.max_len,
                "text_vocab": manifest["text_vocab"],
                "n_colors": len(manifest["palette_rgb"]),
            },
        }
        torch.save(checkpoint, "checkpoint.pt")
        mlflow.log_artifact("checkpoint.pt")

if __name__ == "__main__":
    main()