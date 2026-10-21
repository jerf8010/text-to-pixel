"""Load a trained run (either architecture) from MLflow, ready for evaluation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch

from pixelgen.data.generate import build_spec
from pixelgen.data.shapes import ShapeSpec
from pixelgen.data.tokenizer import TextTokenizer
from pixelgen.evaluation.generators import (
    decoder_only_generator,
    encoder_decoder_generator,
)
from pixelgen.evaluation.validity import GenerateFn
from pixelgen.models.factory import build_model

SIZE_KEYS = ("text_vocab_size", "n_colors", "max_len")
GENERATORS = {
    "decoder_only": decoder_only_generator,
    "encoder_decoder": encoder_decoder_generator,
}


@dataclass
class LoadedRun:
    """Everything needed to evaluate or sample from a trained run."""

    model: torch.nn.Module
    kind: str
    tok: TextTokenizer
    spec: ShapeSpec
    manifest: dict
    data_dir: Path
    sizes: dict
    device: str

    def generate_fn(self, temperature: float = 1.0) -> GenerateFn:
        """Return a ``generate_fn(captions) -> images`` for this run's model."""
        return GENERATORS[self.kind](
            self.model, self.tok, max_len=self.sizes["max_len"],
            n_colors=self.sizes["n_colors"], device=self.device,
            temperature=temperature,
        )


def load_run(run_id: str, device: str) -> LoadedRun:
    """Rebuild the model, tokenizer and data spec of an MLflow run.

    Args:
        run_id: MLflow run id; the run must contain a ``checkpoint.pt`` artifact.
        device: device to load the model on.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    run = mlflow.get_run(run_id)
    data_dir = Path(run.data.params["data_dir"])

    ckpt_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checkpoint.pt"
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    mcfg = dict(ckpt["model_config"])
    sizes = {k: mcfg.pop(k) for k in SIZE_KEYS}
    model = build_model(mcfg, **sizes).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    manifest = json.loads((data_dir / "manifest.json").read_text())
    tok = TextTokenizer(manifest["text_vocab"])
    assert sizes["text_vocab_size"] == tok.vocab_size, "vocabulary mismatch"
    assert sizes["n_colors"] == len(manifest["palette_rgb"]), "palette mismatch"

    return LoadedRun(
        model=model, kind=mcfg["type"], tok=tok, spec=build_spec(manifest["config"]),
        manifest=manifest, data_dir=data_dir, sizes=sizes, device=device,
    )