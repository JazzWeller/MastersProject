"""Transformer building blocks shared by the networks.

**`FastEncoderLayer`** is a drop-in for `nn.TransformerEncoderLayer` as the
networks configure it (pre-LN, GELU, batch-first). It has the same
parameter names -- `self_attn.in_proj_weight`, `self_attn.out_proj`,
`linear1`, `linear2`, `norm1`, `norm2` -- so a state dict moves between the
two unchanged, and computes the same function. What differs is the
attention call: `F.scaled_dot_product_attention` with a chosen kernel
priority, where `nn.MultiheadAttention`'s training path left the kernel to
the dispatcher. Measured on the RTX 5060 Ti (2026-10-05): +3-8% real BC
samples/s and +6-25% inference at 73 tokens, and +45-52% training at 400
tokens (trunk only), where attention is a larger share of the work.

**`Trunk`** stacks layers with `nn.TransformerEncoder`'s state-dict layout
(`layers.<i>.*`), and can recompute each layer's activations in the
backward pass instead of storing them (`checkpoint = True`, the
`activation_checkpointing` switch). Real BC steps at batch 512: peak memory
1/2 to 1/3, for 8-15% fewer samples/s (d=256/8 layers: 2.70 -> 0.97 GiB,
6.7k -> 5.8k). Inference and `no_grad` passes never recompute.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

# Kernel priority per `network.attention_kernel`. Every list ends in MATH,
# so an input a fast kernel can't take (fp32, CPU, an odd head size) falls
# back instead of raising.
KERNELS = {
    "flash": [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    "cudnn": [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    "efficient": [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH],
    "math": [SDPBackend.MATH],
    "auto": None,  # the dispatcher's own choice
}


class SelfAttention(nn.Module):
    """`nn.MultiheadAttention`'s parameters (packed q/k/v input projection,
    output projection) and initialization, computed with SDPA."""

    def __init__(self, d: int, heads: int, dropout: float, kernel: str):
        super().__init__()
        if d % heads:
            raise ValueError(f"d_model {d} is not divisible by heads {heads}")
        if kernel not in KERNELS:
            raise ValueError(f"network.attention_kernel must be one of {sorted(KERNELS)}, not {kernel!r}")
        self.heads = heads
        self.dropout = dropout
        self.kernels: Optional[Sequence[SDPBackend]] = KERNELS[kernel]
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * d))
        self.out_proj = nn.Linear(d, d)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        p = self.dropout if self.training else 0.0
        if self.kernels is None:
            a = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        else:
            with sdpa_kernel(self.kernels, set_priority=True):
                a = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        return self.out_proj(a.transpose(1, 2).reshape(B, T, d))


class FastEncoderLayer(nn.Module):
    """Pre-LN encoder layer: `x + attn(norm1(x))`, then `x + ff(norm2(x))`,
    with dropout where `nn.TransformerEncoderLayer` puts it."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float = 0.0, kernel: str = "flash"):
        super().__init__()
        self.self_attn = SelfAttention(d, heads, dropout, kernel)
        self.linear1 = nn.Linear(d, ff)
        self.linear2 = nn.Linear(ff, d)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.self_attn(self.norm1(x)))
        return x + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x))))))


class Trunk(nn.Module):
    """A stack of layers, `nn.TransformerEncoder`'s state-dict layout."""

    def __init__(self, layers: Sequence[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recompute = self.checkpoint and self.training and torch.is_grad_enabled()
        for layer in self.layers:
            # Non-reentrant checkpointing keeps the RNG state, so dropout
            # masks in the recompute match the forward pass exactly.
            x = checkpoint(layer, x, use_reentrant=False) if recompute else layer(x)
        return x


def build_trunk(d: int, heads: int, ff: int, layers: int, dropout: float, attention: str, kernel: str) -> nn.Module:
    """`attention`: `sdpa` (FastEncoderLayer) or `torch`
    (nn.TransformerEncoderLayer, the layer the v1 network was trained
    with). Both give the same state-dict keys."""
    if attention == "sdpa":
        return Trunk([FastEncoderLayer(d, heads, ff, dropout, kernel) for _ in range(layers)])
    if attention == "torch":
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
    raise ValueError(f"network.attention must be sdpa or torch, not {attention!r}")


def set_activation_checkpointing(trunk: nn.Module, enabled: bool) -> None:
    """The memory-saving switch. Only the `sdpa` trunk supports it."""
    if not enabled:
        if isinstance(trunk, Trunk):
            trunk.checkpoint = False
        return
    if not isinstance(trunk, Trunk):
        raise ValueError("activation_checkpointing needs network.attention = 'sdpa'")
    trunk.checkpoint = True
