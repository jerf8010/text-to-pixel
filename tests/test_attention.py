import pytest
import torch

from pixelgen.models.attention import Head, MultiHeadAttention


def make_head():
    torch.manual_seed(0)
    head = Head(n_embd=32, head_size=16, block_size=16, dropout=0.0)
    head.eval()
    return head


def make_mha():
    torch.manual_seed(0)
    mha = MultiHeadAttention(n_embd=32, n_head=4, block_size=16, dropout=0.0)
    mha.eval()
    return mha


def assert_causal(module, x):
    """Changing only the last token must not change the outputs at earlier positions."""
    x_changed = x.clone()
    x_changed[:, -1, :] = torch.randn(x.shape[0], x.shape[2])
    with torch.no_grad():
        out1, out2 = module(x), module(x_changed)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-6)
    assert not torch.allclose(out1[:, -1], out2[:, -1])


def test_head_output_shape():
    assert make_head()(torch.randn(2, 10, 32)).shape == (2, 10, 16)


def test_head_is_causal():
    assert_causal(make_head(), torch.randn(2, 10, 32))


def test_multihead_keeps_the_shape():
    x = torch.randn(2, 10, 32)
    assert make_mha()(x).shape == x.shape


def test_multihead_is_causal():
    assert_causal(make_mha(), torch.randn(2, 10, 32))


def test_n_embd_must_be_divisible_by_n_head():
    with pytest.raises(AssertionError):
        MultiHeadAttention(n_embd=32, n_head=5, block_size=16, dropout=0.0)