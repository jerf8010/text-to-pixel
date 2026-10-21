"""Extract cross-attention weights from a trained encoder-decoder model."""
from __future__ import annotations

import torch


@torch.no_grad()
def cross_attention_maps(
    model: torch.nn.Module,
    src: torch.Tensor,
    src_pad_mask: torch.Tensor,
    pixels: torch.Tensor,
    n_colors: int,
) -> torch.Tensor:
    """Return the decoder's cross-attention weights for given images.

    The image is fed back with teacher forcing (``[BOS] + pixels[:-1]``), so the
    weights at position ``t`` are those used to predict pixel ``t``.

    Args:
        model: a trained EncoderDecoderTransformer.
        src: caption token ids, shape (B, S).
        src_pad_mask: bool tensor (B, S), True at padding positions.
        pixels: palette indices, shape (B, 16, 16) or (B, 256).
        n_colors: number of palette colors; also the BOS id.

    Returns:
        Tensor of shape (n_layer, n_head, B, T, S), with T = 256 pixel positions.
    """
    model.eval()
    device = next(model.parameters()).device
    src, src_pad_mask = src.to(device), src_pad_mask.to(device)
    flat = pixels.reshape(pixels.shape[0], -1).long().to(device)
    bos = torch.full((flat.shape[0], 1), n_colors, dtype=torch.long, device=device)
    tgt_in = torch.cat([bos, flat[:, :-1]], dim=1)

    heads = [h for blk in model.decoder for h in blk.ca.heads]
    for h in heads:
        h.store_attn = True
    try:
        model(src, tgt_in, src_pad_mask)
        per_layer = [
            torch.stack([h.last_attn for h in blk.ca.heads]) for blk in model.decoder
        ]  # each (n_head, B, T, S)
    finally:
        for h in heads:
            h.store_attn = False
            h.last_attn = None
    return torch.stack(per_layer).cpu()