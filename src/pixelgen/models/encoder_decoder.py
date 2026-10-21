from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixelgen.models.blocks import DecoderBlock, EncoderBlock

class EncoderDecoderTransformer(nn.Module):
    """Encoder-decoder Transformer: text (source) -> pixels (target).

    The encoder reads the text bidirectionally and produces a memory. The decoder
    generates pixels autoregressively, attending to the memory via cross-attention.

    Args:
        src_vocab_size: number of text token ids (including pad/unk).
        tgt_vocab_size: number of pixel token ids (colors plus BOS).
        n_embd: embedding size.
        n_head: attention heads per block; ``n_embd`` must be divisible by it.
        n_layer: number of blocks in the encoder and in the decoder.
        max_src_len: maximum text length.
        max_tgt_len: maximum pixel sequence length.
        dropout: dropout probability.
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, n_embd, n_head, n_layer,
                 max_src_len, max_tgt_len, dropout):
        super().__init__()
        self.src_tok = nn.Embedding(src_vocab_size, n_embd)
        self.src_pos = nn.Embedding(max_src_len, n_embd)
        self.encoder = nn.ModuleList(
            [EncoderBlock(n_embd, n_head, max_src_len, dropout) for _ in range(n_layer)]
        )
        self.enc_ln = nn.LayerNorm(n_embd)
        self.tgt_tok = nn.Embedding(tgt_vocab_size, n_embd)
        self.tgt_pos = nn.Embedding(max_tgt_len, n_embd)
        self.decoder = nn.ModuleList(
            [DecoderBlock(n_embd, n_head, max_tgt_len, dropout) for _ in range(n_layer)]
        )
        self.dec_ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, tgt_vocab_size)

    def encode(self, src, src_pad_mask):
        S = src.shape[1]
        x = self.src_tok(src) + self.src_pos(torch.arange(S, device=src.device))
        for blk in self.encoder:
            x = blk(x, pad_mask=src_pad_mask)
        return self.enc_ln(x)
    
    def decode(self, tgt_in, memory, src_pad_mask):
        T = tgt_in.shape[1]
        x = self.tgt_tok(tgt_in) + self.tgt_pos(torch.arange(T, device=tgt_in.device))
        for blk in self.decoder:
            x = blk(x, memory, src_pad_mask)
        return self.head(self.dec_ln(x))
    
    def forward(self, src, tgt_in, src_pad_mask, targets=None):
        memory = self.encode(src, src_pad_mask)
        logits = self.decode(tgt_in, memory, src_pad_mask)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))        
        return logits, loss