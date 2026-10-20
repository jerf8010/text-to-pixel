"""Attention building blocks: causal self-attention, bidirectional self-attention and cross-attention."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of scaled dot-product attention.

    Queries are computed from ``x``; keys and values are computed from ``context``
    (``x`` itself when no context is given). This single class covers three cases:

    * causal self-attention (decoder): ``causal=True``, no context;
    * bidirectional self-attention (encoder): ``causal=False``, no context;
    * cross-attention (decoder -> encoder): ``causal=False``, ``context`` = encoder output.

    With ``causal=True``, position ``t`` attends only to positions ``0..t``, so its
    output never depends on later tokens.

    Args:
        n_embd: size of the input embeddings.
        head_size: size of the keys, queries and values (and of the output).
        block_size: maximum sequence length; sets the size of the causal mask.
        dropout: dropout probability applied to the attention weights.
        causal: if True, hide future positions (requires ``context`` length == ``x`` length).
    """

    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.causal = causal
        # Lower-triangular matrix: position t may look at positions 0..t only.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention.

        Args:
            x: queries source, shape (B, T, n_embd).
            context: keys/values source, shape (B, S, n_embd); defaults to ``x``.
            pad_mask: bool tensor of shape (B, S), True where the context token is
                padding and must be ignored.

        Returns:
            Tensor of shape (B, T, head_size).
        """
        _, T, _ = x.shape
        if context is None:
            context = x

        S = context.shape[1]
        if self.causal:
            assert S == T, "causal attention requires S == T"
        k = self.key(context)  # (B, S, hs)
        q = self.query(x)  # (B, T, hs)

        # Attention scores, scaled by sqrt(head_size) to keep the softmax well behaved.
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, S)

        if self.causal:
            wei = wei.masked_fill(self.tril[:T, :S] == 0, float("-inf"))  # hide the future
        if pad_mask is not None:
            wei = wei.masked_fill(pad_mask[:, None, :], float("-inf"))  # hide padding

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(context)  # (B, S, hs)
        return wei @ v  # weighted sum of values: (B, T, hs)


class MultiHeadAttention(nn.Module):
    """Several attention heads running in parallel.

    The head outputs are concatenated and mixed by a linear projection. Supports
    causal self-attention, bidirectional self-attention and cross-attention (see
    :class:`Head`).

    Args:
        n_embd: size of the input and output embeddings.
        n_head: number of heads; ``n_embd`` must be divisible by it.
        block_size: maximum sequence length.
        dropout: dropout probability.
        causal: if True, hide future positions.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        causal: bool = True,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout, causal=causal) for _ in range(n_head)]
        )
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply multi-head attention.

        Args:
            x: queries source, shape (B, T, n_embd).
            context: keys/values source, shape (B, S, n_embd); defaults to ``x``.
            pad_mask: bool tensor (B, S), True where the context token is padding.

        Returns:
            Tensor of shape (B, T, n_embd).
        """
        out = torch.cat([head(x, context, pad_mask) for head in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))