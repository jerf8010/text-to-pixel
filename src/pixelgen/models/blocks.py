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
    
class EncoderBlock(nn.Module):
    """Pre-LN Transformer encoder block: bidirectional self-attention, then feed-forward.

    Args:
        n_embd: size of the embeddings.
        n_head: number of attention heads; ``n_embd`` must be divisible by it.
        block_size: maximum source (text) length.
        dropout: dropout probability.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout, causal=False)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Map ``x`` (B, S, n_embd) to the same shape; ``pad_mask`` (B, S) is True at padding."""
        x = x + self.sa(self.ln1(x), pad_mask=pad_mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    """Pre-LN Transformer decoder block with three sub-layers.

    Causal self-attention over the generated tokens, cross-attention to the encoder
    output (``memory``), and a position-wise feed-forward network.

    Args:
        n_embd: size of the embeddings.
        n_head: number of attention heads; ``n_embd`` must be divisible by it.
        block_size: maximum target (pixel) length.
        dropout: dropout probability.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout, causal=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ca = MultiHeadAttention(n_embd, n_head, block_size, dropout, causal=False)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map ``x`` (B, T, n_embd) to the same shape.

        ``memory`` is the encoder output (B, S, n_embd); ``memory_pad_mask`` (B, S)
        is True where the source token is padding.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ca(self.ln2(x), context=memory, pad_mask=memory_pad_mask)
        x = x + self.ffwd(self.ln3(x))
        return x