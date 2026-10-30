"""A tiny byte-level decoder. Real autograd and LoRA, deliberately no model downloads."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

PAD, BOS, EOS = 256, 257, 258
VOCAB = 259


@dataclass(frozen=True)
class ModelConfig:
    width: int = 48
    heads: int = 4
    layers: int = 2
    context: int = 96
    rank: int = 4
    alpha: float = 8.0

    def __post_init__(self):
        if self.width < 4 or self.heads < 1 or self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.layers < 1 or self.context < 8 or not 1 <= self.rank <= self.width:
            raise ValueError("invalid layer/context/rank configuration")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be positive and finite")


class LoRALinear(nn.Module):
    def __init__(self, inputs: int, outputs: int, rank: int, alpha: float):
        super().__init__()
        self.base = nn.Linear(inputs, outputs, bias=False)
        self.lora_a = nn.Parameter(torch.randn(rank, inputs) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(outputs, rank))
        self.scale = alpha / rank
        self.enabled = False

    def forward(self, x):
        y = self.base(x)
        if self.enabled:
            y = y + F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale
        return y


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        d, r, a = config.width, config.rank, config.alpha
        self.heads = config.heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = LoRALinear(d, 3 * d, r, a)
        self.proj = LoRALinear(d, d, r, a)
        self.up = LoRALinear(d, 4 * d, r, a)
        self.down = LoRALinear(4 * d, d, r, a)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        x = x + self.proj(attended.transpose(1, 2).reshape(b, t, d))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class TinyLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(VOCAB, config.width)
        self.positions = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.LayerNorm(config.width)
        self.head = LoRALinear(config.width, VOCAB, config.rank, config.alpha)

    def forward(self, ids):
        if ids.ndim != 2 or ids.shape[1] > self.config.context:
            raise ValueError("expected [batch, time] within context limit")
        x = self.tokens(ids) + self.positions(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def mode(self, adapter: bool) -> None:
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.enabled = adapter
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(is_adapter(name) if adapter else not is_adapter(name))

    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 32) -> str:
        ids = [BOS, *prompt.encode("utf-8"), 10]
        if len(ids) + max_new > self.config.context:
            raise ValueError("generation exceeds context limit")
        generated = []
        for _ in range(max_new):
            next_id = int(self(torch.tensor([ids]))[0, -1].argmax())
            if next_id == EOS:
                break
            ids.append(next_id)
            generated.append(next_id)
        return bytes(i for i in generated if i < 256).decode("utf-8", errors="replace")


def is_adapter(name: str) -> bool:
    return name.endswith(("lora_a", "lora_b"))


def parameter_hash(model: TinyLM, *, adapter: bool) -> str:
    h = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if is_adapter(name) == adapter:
            h.update(name.encode())
            h.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encode_batch(pairs: list[tuple[str, str]], context: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not pairs:
        raise ValueError("batch must not be empty")
    encoded = []
    for prompt, response in pairs:
        if not response:
            raise ValueError("response must not be empty")
        prefix = [BOS, *prompt.encode("utf-8"), 10]
        sequence = prefix + list(response.encode("utf-8")) + [EOS]
        if len(sequence) - 1 > context:
            raise ValueError("example exceeds context; truncation is never silent")
        # labels[t] predicts sequence[t+1]. The last prompt position predicts
        # the FIRST response byte, so it must not be masked.
        labels = [-100] * (len(prefix) - 1) + sequence[len(prefix) :]
        encoded.append((sequence[:-1], labels))
    longest = max(len(x) for x, _ in encoded)
    x = torch.full((len(pairs), longest), PAD, dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (ids, labels) in enumerate(encoded):
        x[i, : len(ids)] = torch.tensor(ids)
        y[i, : len(labels)] = torch.tensor(labels)
    return x, y


def response_loss(model: TinyLM, pairs: list[tuple[str, str]]) -> torch.Tensor:
    x, y = encode_batch(pairs, model.config.context)
    return F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
