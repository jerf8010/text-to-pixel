import math

import pytest
import torch
import torch.nn.functional as F

from pixelgen.models.decoder_only import DecoderOnlyTransformer

VOCAB, BLOCK = 16, 16


def make_model(**overrides):
    torch.manual_seed(0)
    cfg = dict(vocab_size=VOCAB, n_embd=32, n_head=4, n_layer=2, block_size=BLOCK, dropout=0.0)
    cfg.update(overrides)
    return DecoderOnlyTransformer(**cfg).eval()


def random_batch(batch=4, length=12):
    idx = torch.randint(0, VOCAB, (batch, length))
    targets = torch.randint(0, VOCAB, (batch, length))
    return idx, targets


def test_logits_shape_and_no_loss_without_targets():
    idx, _ = random_batch()
    logits, loss = make_model()(idx)
    assert logits.shape == (4, 12, VOCAB)
    assert loss is None


def test_initial_loss_is_close_to_uniform_guessing():
    torch.manual_seed(1)
    idx, targets = random_batch(batch=16)
    _, loss = make_model()(idx, targets)
    assert loss.ndim == 0
    assert abs(loss.item() - math.log(VOCAB)) < 0.3  # an untrained model guesses uniformly


def test_ignored_targets_do_not_count():
    idx, targets = random_batch()
    half = 6
    masked = targets.clone()
    masked[:, :half] = -100
    model = make_model()
    logits, loss = model(idx, masked)
    expected = F.cross_entropy(
        logits[:, half:].reshape(-1, VOCAB), targets[:, half:].reshape(-1)
    )
    assert torch.allclose(loss, expected, atol=1e-6)


def test_model_is_causal():
    model = make_model()
    idx = torch.randint(0, VOCAB, (2, 10))
    changed = idx.clone()
    changed[:, -1] = (changed[:, -1] + 1) % VOCAB
    with torch.no_grad():
        logits1, _ = model(idx)
        logits2, _ = model(changed)
    assert torch.allclose(logits1[:, :-1], logits2[:, :-1], atol=1e-5)
    assert not torch.allclose(logits1[:, -1], logits2[:, -1])


def test_sequence_longer_than_block_size_is_rejected():
    idx = torch.randint(0, VOCAB, (1, BLOCK + 1))
    with pytest.raises(AssertionError):
        make_model()(idx)


def test_all_parameters_receive_gradients():
    idx, targets = random_batch()
    model = make_model()
    _, loss = model(idx, targets)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, name


def test_can_overfit_a_tiny_batch():
    torch.manual_seed(0)
    seq = torch.randint(0, VOCAB, (4, 13))
    idx, targets = seq[:, :-1], seq[:, 1:]  # next-token prediction on fixed sequences
    model = make_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    _, first_loss = model(idx, targets)
    for _ in range(300):
        _, loss = model(idx, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.5 * first_loss.item()