"""Explicit decoder-only transformer: no pretrained model or wrapper."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int = 128
    embedding_width: int = 128
    heads: int = 4
    layers: int = 4

    def __post_init__(self):
        if min(self.vocab_size, self.context_length, self.embedding_width, self.heads, self.layers) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.embedding_width % self.heads:
            raise ValueError("Embedding width must divide evenly into heads")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.embedding_width
        self.heads = config.heads
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.register_buffer("mask", torch.tril(torch.ones(
            config.context_length, config.context_length, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        batch_size, length, width = x.shape
        # Each head gets its own query, key, and value vectors.
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch_size, length, self.heads, width // self.heads)
                   .transpose(1, 2) for t in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(width // self.heads)
        # Above-diagonal entries refer to future characters. -inf makes their
        # softmax probabilities exactly zero, so no future value can contribute.
        scores = scores.masked_fill(~self.mask[:length, :length], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        mixed = (weights @ v).transpose(1, 2).contiguous().reshape(batch_size, length, width)
        return self.projection(mixed)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.embedding_width
        self.attention_norm = nn.LayerNorm(width)
        self.attention = CausalSelfAttention(config)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(),
                                         nn.Linear(4 * width, width))

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feedforward(self.feedforward_norm(x))


class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_width)
        self.position_embedding = nn.Embedding(config.context_length, config.embedding_width)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.embedding_width)
        self.output = nn.Linear(config.embedding_width, config.vocab_size)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        length = tokens.shape[1]
        if not 1 <= length <= self.config.context_length:
            raise ValueError("Sequence length must be within the context window")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        logits = self.output(self.final_norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss
