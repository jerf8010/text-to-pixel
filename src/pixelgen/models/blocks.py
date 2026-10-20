"""Transformer block: attention followed by a position-wise feed-forward network."""
from __future__ import annotations

import torch
import torch.nn as nn

from pixelgen.models.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> ReLU -> Linear -> Dropout.

    The hidden layer is four times wider than the embedding, as in the original
    Transformer. It is applied independently to every position.

    Args:
        n_embd: size of the input and output embeddings.
        dropout: dropout probability.
    """

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``x`` of shape (B, T, n_embd) to the same shape."""
        return self.net(x)


class Block(nn.Module):
    """Causal Transformer block: communication (attention) then computation (feed-forward).

    Uses pre-layer-norm and residual connections: each sub-layer normalizes its input and
    its output is added back to the input, which keeps deep stacks easy to train.

    Args:
        n_embd: size of the embeddings.
        n_head: number of attention heads; ``n_embd`` must be divisible by it.
        block_size: maximum sequence length.
        dropout: dropout probability.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``x`` of shape (B, T, n_embd) to the same shape."""
        x = x + self.sa(self.ln1(x))  # tokens exchange information
        x = x + self.ffwd(self.ln2(x))  # each token is processed on its own
        return x