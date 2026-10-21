"""Plot, for each word of a caption, where in the generated image it is attended to.

Usage (from the project root; the run must be an encoder-decoder):
    python scripts/attention_figure.py --run-id <run_id> --index 3
    python scripts/attention_figure.py --run-id <run_id> --caption "a small red circle left of a large blue square"
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pixelgen.data.shapes import CANVAS
from pixelgen.evaluation.attention_maps import cross_attention_maps
from pixelgen.evaluation.loading import load_run
from pixelgen.evaluation.sampling import generate_images_encdec
from pixelgen.training.loop import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="test_comp")
    parser.add_argument("--index", type=int, default=0, help="caption index in the split")
    parser.add_argument("--caption", default=None, help="custom caption (overrides --index)")
    parser.add_argument("--layer", default="mean", help="decoder layer index or 'mean'")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = get_device()
    loaded = load_run(args.run_id, device)
    if loaded.kind != "encoder_decoder":
        raise SystemExit("cross-attention maps need an encoder_decoder run")
    tok, sizes = loaded.tok, loaded.sizes
    palette = np.array(loaded.manifest["palette_rgb"], dtype=np.uint8)

    if args.caption is not None:
        caption = args.caption
    else:
        arrays = dict(np.load(loaded.data_dir / f"{args.split}.npz"))
        caption = str(arrays["captions"][args.index])

    # Tokenize, sample one image, then replay it to read the attention weights.
    src = torch.tensor([tok.encode(caption, sizes["max_len"])], dtype=torch.long)
    pad = src == tok.pad_id
    torch.manual_seed(args.seed)
    image = generate_images_encdec(
        loaded.model, src, pad, sizes["n_colors"], device, args.temperature
    )  # (1, 16, 16) uint8
    maps = cross_attention_maps(
        loaded.model, src, pad, torch.from_numpy(image), sizes["n_colors"]
    )  # (n_layer, n_head, 1, 256, S)

    if args.layer == "mean":
        attn = maps.mean(dim=(0, 1))[0]  # average over layers and heads -> (256, S)
    else:
        attn = maps[int(args.layer)].mean(dim=0)[0]  # one layer, mean over heads

    words = caption.split()
    fig, axes = plt.subplots(
        1, len(words) + 1, figsize=(1.9 * (len(words) + 1), 2.6), squeeze=False
    )
    axes = axes[0]
    axes[0].imshow(palette[image[0]], interpolation="nearest")
    axes[0].set_title("generated", fontsize=8)
    for w, word in enumerate(words):
        heat = attn[:, w].reshape(CANVAS, CANVAS).numpy()
        axes[w + 1].imshow(heat, cmap="magma", interpolation="nearest", vmin=0)
        axes[w + 1].set_title(word, fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"cross-attention ({args.layer}) | \"{caption}\"", fontsize=9)
    fig.tight_layout()
    out = Path(args.out or f"reports/attention_{args.run_id[:8]}_{args.split}_{args.index}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()