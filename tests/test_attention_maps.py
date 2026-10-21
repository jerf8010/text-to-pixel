import torch

from pixelgen.evaluation.attention_maps import cross_attention_maps
from pixelgen.models.encoder_decoder import EncoderDecoderTransformer


def test_cross_attention_maps_shape_and_normalisation():
    torch.manual_seed(0)
    model = EncoderDecoderTransformer(
        src_vocab_size=29, tgt_vocab_size=10, n_embd=32, n_head=4,
        n_layer=2, max_src_len=10, max_tgt_len=256, dropout=0.0,
    )
    src = torch.randint(0, 29, (2, 10))
    pad = torch.zeros(2, 10, dtype=torch.bool)
    pad[:, -3:] = True
    pixels = torch.randint(0, 9, (2, 16, 16))

    maps = cross_attention_maps(model, src, pad, pixels, n_colors=9)

    assert maps.shape == (2, 4, 2, 256, 10)  # layers, heads, batch, pixels, words
    assert torch.allclose(maps.sum(-1), torch.ones(2, 4, 2, 256), atol=1e-5)
    assert maps[..., -3:].abs().max() == 0  # padding gets no attention
    # Storing is switched off again afterwards.
    assert all(not h.store_attn for blk in model.decoder for h in blk.ca.heads)