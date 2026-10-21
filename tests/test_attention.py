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

def test_cross_attention_shape():
    mha = MultiHeadAttention(n_embd=32, n_head=4, block_size=16, dropout=0.0, causal=False)
    x = torch.randn(2, 6, 32)
    context = torch.randn(2, 4, 32)
    assert mha(x, context=context).shape == (2, 6, 32)

def test_pad_mask_ignores_padding():
    torch.manual_seed(0)
    mha = MultiHeadAttention(n_embd=32, n_head=4, block_size=16, dropout=0.0, causal=False)
    x = torch.randn(2, 6, 32)
    context = torch.randn(2, 4, 32)
    pad_mask = torch.zeros(2, 4, dtype=torch.bool)
    pad_mask[:, -1] = True  # last context position is padding

    out1 = mha(x, context=context, pad_mask=pad_mask)

    context2 = context.clone()
    context2[:, -1] = torch.randn(2, 32)  # change only the padded position
    out2 = mha(x, context=context2, pad_mask=pad_mask)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_non_causal_sees_future():
    torch.manual_seed(0)
    mha = MultiHeadAttention(n_embd=32, n_head=4, block_size=16, dropout=0.0, causal=False)
    x = torch.randn(1, 5, 32)
    out1 = mha(x)

    x2 = x.clone()
    x2[:, -1] = torch.randn(1, 32)  # change the last token only
    out2 = mha(x2)

    assert not torch.allclose(out1[:, 0], out2[:, 0], atol=1e-6)


def test_causal_does_not_see_future():
    torch.manual_seed(0)
    mha = MultiHeadAttention(n_embd=32, n_head=4, block_size=16, dropout=0.0, causal=True)
    x = torch.randn(1, 5, 32)
    out1 = mha(x)

    x2 = x.clone()
    x2[:, -1] = torch.randn(1, 32)
    out2 = mha(x2)

    assert torch.allclose(out1[:, 0], out2[:, 0], atol=1e-6)