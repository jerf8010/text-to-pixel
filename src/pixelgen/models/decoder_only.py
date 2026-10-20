"""Decoder-only Transformer: predicts the next token in a single mixed sequence."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixelgen.models.blocks import Block


class DecoderOnlyTransformer(nn.Module):
    """Causal Transformer language model over a single shared vocabulary.

    Used here as the baseline: caption tokens, a separator and the image pixels all live
    in one sequence, and the model predicts the next token at every position.

    Args:
        vocab_size: number of distinct token ids (text, separator and colors).
        n_embd: size of the embeddings.
        n_head: number of attention heads; ``n_embd`` must be divisible by it.
        n_layer: number of stacked Transformer blocks.
        block_size: maximum sequence length the model can process.
        dropout: dropout probability.
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights from N(0, 0.02) and biases with zeros."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute next-token logits and, if ``targets`` is given, the loss.

        Args:
            idx: token ids of shape (B, T), with T <= block_size.
            targets: next-token ids of shape (B, T). Positions set to -100 are ignored.

        Returns:
            logits of shape (B, T, vocab_size) and the cross-entropy loss (or None).
        """
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} exceeds block_size {self.block_size}"

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        # cross_entropy expects (N, classes) and (N,); -100 targets are ignored by default.
        loss = F.cross_entropy(logits.reshape(B * T, -1), targets.reshape(B * T))
        return logits, loss