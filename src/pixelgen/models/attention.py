"""Causal self-attention building blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of causal self-attention.

    Each position attends only to itself and to earlier positions, so the output at
    position ``t`` never depends on tokens after ``t``.

    Args:
        n_embd: size of the input embeddings.
        head_size: size of the keys, queries and values (and of the output).
        block_size: maximum sequence length; sets the size of the causal mask.
        dropout: dropout probability applied to the attention weights.
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Lower-triangular matrix: position t may look at positions 0..t only.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention to ``x`` of shape (B, T, n_embd); returns (B, T, head_size)."""
        _, T, _ = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)

        # Attention scores, scaled by sqrt(head_size) to keep the softmax well behaved.
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # hide the future
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # (B, T, hs)
        return wei @ v  # weighted sum of values: (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several causal attention heads running in parallel.

    The head outputs are concatenated and mixed by a linear projection.

    Args:
        n_embd: size of the input and output embeddings.
        n_head: number of heads; ``n_embd`` must be divisible by it.
        block_size: maximum sequence length.
        dropout: dropout probability.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention to ``x`` of shape (B, T, n_embd); returns the same shape."""
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))