"""Model factory: build either architecture from a config dict."""
from __future__ import annotations

from pixelgen.models.decoder_only import DecoderOnlyTransformer
from pixelgen.models.encoder_decoder import EncoderDecoderTransformer


def build_model(
    mcfg: dict, *, text_vocab_size: int, n_colors: int, max_len: int, n_pixels: int = 256
):
    """Build a model from ``mcfg`` (which must contain ``type``).

    Args:
        mcfg: model hyperparameters (n_embd, n_head, n_layer, dropout) plus ``type``.
        text_vocab_size: number of text token ids.
        n_colors: number of palette colors (the BOS id equals ``n_colors``).
        max_len: maximum caption length in tokens.
        n_pixels: number of pixels per image.
    """
    params = {k: v for k, v in mcfg.items() if k != "type"}
    kind = mcfg["type"]
    if kind == "decoder_only":
        return DecoderOnlyTransformer(
            vocab_size=text_vocab_size + 1 + n_colors,
            block_size=max_len + n_pixels,
            **params,
        )
    if kind == "encoder_decoder":
        return EncoderDecoderTransformer(
            src_vocab_size=text_vocab_size,
            tgt_vocab_size=n_colors + 1,
            max_src_len=max_len,
            max_tgt_len=n_pixels,
            **params,
        )
    raise ValueError(f"unknown model type: {kind}")