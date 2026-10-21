"""Save a grid of generated images next to reference images, marking correct ones.

Usage (from the project root):
    python scripts/sample_grid.py --run-id <run_id> --split test_comp
    python scripts/sample_grid.py --run-id <run_id> --failures-only
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pixelgen.evaluation.loading import load_run
from pixelgen.evaluation.metrics import scene_from_row, valid_renders
from pixelgen.training.loop import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="test_comp")
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    loaded = load_run(args.run_id, get_device())
    arrays = dict(np.load(loaded.data_dir / f"{args.split}.npz"))
    palette = np.array(loaded.manifest["palette_rgb"], dtype=np.uint8)
    generate = loaded.generate_fn(args.temperature)

    # With --failures-only, draw a larger pool so enough failures are available.
    pool = args.n * (10 if args.failures_only else 1)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(arrays["images"]), size=min(pool, len(arrays["images"])),
                         replace=False)
    captions = [str(arrays["captions"][i]) for i in indices]

    torch.manual_seed(args.seed)
    images = np.concatenate(
        [generate(captions[s:s + 100]) for s in range(0, len(captions), 100)]
    )
    correct = np.array([
        img.tobytes() in valid_renders(scene_from_row(arrays, int(i)), loaded.spec)
        for i, img in zip(indices, images)
    ])

    keep = np.flatnonzero(~correct if args.failures_only else np.ones_like(correct))[: args.n]
    cols = 4  # (generated, reference) pairs per row
    rows = int(np.ceil(len(keep) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for k, j in enumerate(keep):
        r, c = divmod(k, cols)
        gen_ax, ref_ax = axes[r, 2 * c], axes[r, 2 * c + 1]
        gen_ax.imshow(palette[images[j]], interpolation="nearest")
        ref_ax.imshow(palette[arrays["images"][indices[j]]], interpolation="nearest")
        gen_ax.set_title(textwrap.fill(captions[j], 22), fontsize=6.5)
        ref_ax.set_title("reference", fontsize=6.5, color="gray")
        color = "tab:green" if correct[j] else "tab:red"
        gen_ax.set_xlabel("correct" if correct[j] else "wrong", fontsize=7, color=color)
        gen_ax.xaxis.label.set_visible(True)
        for spine in gen_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        gen_ax.set_xticks([])
        gen_ax.set_yticks([])
        gen_ax.axis("on")

    rate = correct.mean()
    fig.suptitle(
        f"{loaded.kind} | {args.split} | valid rate in this pool: {rate:.2f}", fontsize=10
    )
    fig.tight_layout()
    suffix = "failures" if args.failures_only else "samples"
    out = Path(args.out or f"reports/{suffix}_{loaded.kind}_{args.run_id[:8]}_{args.split}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()