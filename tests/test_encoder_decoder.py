import torch
from pixelgen.models.encoder_decoder import EncoderDecoderTransformer


def make_model():
    return EncoderDecoderTransformer(
        src_vocab_size=29, tgt_vocab_size=10, n_embd=32, n_head=4,
        n_layer=2, max_src_len=10, max_tgt_len=256, dropout=0.0,
    ).eval()


def make_batch(B=2):
    src = torch.randint(0, 29, (B, 10))
    pad = torch.zeros(B, 10, dtype=torch.bool)
    pad[:, -3:] = True
    tgt_in = torch.randint(0, 10, (B, 256))
    targets = torch.randint(0, 9, (B, 256))
    return src, pad, tgt_in, targets


def test_forward_shapes_and_loss():
    model = make_model()
    src, pad, tgt_in, targets = make_batch()
    logits, loss = model(src, tgt_in, pad, targets)
    assert logits.shape == (2, 256, 10)
    assert loss.ndim == 0 and loss.item() > 0
    _, no_loss = model(src, tgt_in, pad)
    assert no_loss is None


def test_initial_loss_near_log_vocab():
    torch.manual_seed(0)
    model = make_model()
    src, pad, tgt_in, targets = make_batch()
    _, loss = model(src, tgt_in, pad, targets)
    assert abs(loss.item() - torch.log(torch.tensor(10.0)).item()) < 1.0


def test_decoder_is_causal():
    model = make_model()
    src, pad, tgt_in, _ = make_batch()
    out1, _ = model(src, tgt_in, pad)
    tgt2 = tgt_in.clone()
    tgt2[:, -1] = (tgt2[:, -1] + 1) % 10
    out2, _ = model(src, tgt2, pad)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)


def test_padding_in_source_is_ignored():
    model = make_model()
    src, pad, tgt_in, _ = make_batch()
    out1, _ = model(src, tgt_in, pad)
    src2 = src.clone()
    src2[:, -3:] = torch.randint(0, 29, (2, 3))
    out2, _ = model(src2, tgt_in, pad)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_text_changes_the_image_logits():
    model = make_model()
    src, pad, tgt_in, _ = make_batch()
    out1, _ = model(src, tgt_in, pad)
    src2 = src.clone()
    src2[:, 0] = (src2[:, 0] + 1) % 29
    out2, _ = model(src2, tgt_in, pad)
    assert not torch.allclose(out1, out2, atol=1e-5)