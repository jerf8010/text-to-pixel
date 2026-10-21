import torch

from pixelgen.models.blocks import Block, FeedForward, EncoderBlock, DecoderBlock


def test_feedforward_keeps_the_shape():
    ffwd = FeedForward(n_embd=32, dropout=0.0)
    x = torch.randn(2, 10, 32)
    assert ffwd(x).shape == x.shape


def test_feedforward_acts_on_each_position_independently():
    torch.manual_seed(0)
    ffwd = FeedForward(n_embd=32, dropout=0.0).eval()
    x = torch.randn(2, 10, 32)
    x_changed = x.clone()
    x_changed[:, 3, :] = torch.randn(2, 32)  # change a single position
    with torch.no_grad():
        out1, out2 = ffwd(x), ffwd(x_changed)
    assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-6)
    assert torch.allclose(out1[:, 4:], out2[:, 4:], atol=1e-6)
    assert not torch.allclose(out1[:, 3], out2[:, 3])


def test_block_keeps_the_shape():
    block = Block(n_embd=32, n_head=4, block_size=16, dropout=0.0)
    x = torch.randn(2, 10, 32)
    assert block(x).shape == x.shape


def test_block_is_causal():
    torch.manual_seed(0)
    block = Block(n_embd=32, n_head=4, block_size=16, dropout=0.0).eval()
    x = torch.randn(2, 10, 32)
    x_changed = x.clone()
    x_changed[:, -1, :] = torch.randn(2, 32)
    with torch.no_grad():
        out1, out2 = block(x), block(x_changed)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-6)
    assert not torch.allclose(out1[:, -1], out2[:, -1])


def test_encoder_block_shape_and_padding():
    torch.manual_seed(0)
    blk = EncoderBlock(n_embd=32, n_head=4, block_size=10, dropout=0.0)
    x = torch.randn(2, 10, 32)
    pad_mask = torch.zeros(2, 10, dtype=torch.bool)
    pad_mask[:, -3:] = True

    out1 = blk(x, pad_mask=pad_mask)
    assert out1.shape == (2, 10, 32)

    # Changing padded positions must not affect real positions.
    x2 = x.clone()
    x2[:, -3:] = torch.randn(2, 3, 32)
    out2 = blk(x2, pad_mask=pad_mask)
    assert torch.allclose(out1[:, :-3], out2[:, :-3], atol=1e-6)

def test_decoder_block_shape_causal_and_memory_padding():
    torch.manual_seed(0)
    blk = DecoderBlock(n_embd=32, n_head=4, block_size=16, dropout=0.0)
    x = torch.randn(2, 16, 32)
    memory = torch.randn(2, 10, 32)
    pad = torch.zeros(2, 10, dtype=torch.bool)
    pad[:, -3:] = True

    out1 = blk(x, memory, pad)
    assert out1.shape == (2, 16, 32)

    # Padded memory positions must not matter.
    m2 = memory.clone()
    m2[:, -3:] = torch.randn(2, 3, 32)
    assert torch.allclose(out1, blk(x, m2, pad), atol=1e-6)

    # Causality: changing the last pixel must not change earlier outputs.
    x2 = x.clone()
    x2[:, -1] = torch.randn(2, 32)
    assert torch.allclose(out1[:, :-1], blk(x2, memory, pad)[:, :-1], atol=1e-6)