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

**v2's attention** (`AttnSpec`, Agent Observation Plan O7/O8) never builds a
[B, T, T] mask: padding is a key mask [B, 1, 1, T], and `stream`'s
block-causal history is a second, causal attention call over the history
rows alone. Its per-layer keys and values can be returned (`want_kv`) and
passed back as `past`: a search's history prefix is then computed once and
each leaf runs only its state tokens and its own suffix rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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


@dataclass
class AttnSpec:
    """How v2's tokens may attend (no [B, T, T] mask is ever built).

    - `key_valid` [B, P + T]: the keys that exist, `past` keys first.
    - `hist_from`: `stream` only -- queries from this index on are history
      rows, which attend causally to history alone (the cached prefix's
      `past` keys, then the rows before them); the queries before it attend
      to every valid key.
    - `past_len` (P): cached prefix keys prepended at every layer.
    - `hist_mask` [B, 1, S, P + S]: with `past`, which keys each of the S
      history queries may see (the prefix's real rows, then its own
      suffix causally)."""

    key_valid: torch.Tensor
    hist_from: Optional[int] = None
    past_len: int = 0
    hist_mask: Optional[torch.Tensor] = None


KV = Tuple[torch.Tensor, torch.Tensor]  # [B, heads, rows, d / heads] each


def _attend(q, k, v, mask, p: float):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)


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

    def forward(self, x: torch.Tensor, spec: Optional[AttnSpec] = None, past: Optional[KV] = None,
                want_kv: bool = False):
        """`spec` None: full attention (v1). Otherwise v2's (`AttnSpec`);
        `past`: this layer's cached prefix keys and values; `want_kv`: also
        return this call's own (k, v)."""
        B, T, d = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.view(B, T, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        p = self.dropout if self.training else 0.0
        if spec is None:
            if self.kernels is None:
                a = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
            else:
                with sdpa_kernel(self.kernels, set_priority=True):
                    a = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        else:
            K, V = (k, v) if past is None else (torch.cat([past[0], k], 2), torch.cat([past[1], v], 2))
            kmask = spec.key_valid[:, None, None, :]
            s = spec.hist_from
            if s is None:
                a = _attend(q, K, V, kmask, p)
            else:
                parts = []
                if s > 0:
                    parts.append(_attend(q[:, :, :s], K, V, kmask, p))
                if s < T:
                    if past is None:  # history rows end each item's sequence: causal is exact
                        parts.append(F.scaled_dot_product_attention(q[:, :, s:], k[:, :, s:], v[:, :, s:],
                                                                    is_causal=True, dropout_p=p))
                    else:
                        kh = torch.cat([past[0], k[:, :, s:]], 2)
                        vh = torch.cat([past[1], v[:, :, s:]], 2)
                        parts.append(_attend(q[:, :, s:], kh, vh, spec.hist_mask, p))
                a = parts[0] if len(parts) == 1 else torch.cat(parts, 2)
        out = self.out_proj(a.transpose(1, 2).reshape(B, T, d))
        return (out, (k, v)) if want_kv else out


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

    def forward(self, x: torch.Tensor, spec: Optional[AttnSpec] = None, past: Optional[KV] = None,
                want_kv: bool = False):
        a = self.self_attn(self.norm1(x), spec, past, want_kv)
        if want_kv:
            a, kv = a
        x = x + self.dropout1(a)
        x = x + self.dropout2(self.linear2(self.dropout(F.gelu(self.linear1(self.norm2(x))))))
        return (x, kv) if want_kv else x


class Trunk(nn.Module):
    """A stack of layers, `nn.TransformerEncoder`'s state-dict layout."""

    def __init__(self, layers: Sequence[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint = False

    def forward(self, x: torch.Tensor, spec: Optional[AttnSpec] = None, past: Optional[List[KV]] = None,
                want_kv: bool = False):
        """`past`: one (k, v) per layer (a cached prefix); `want_kv`: also
        return each layer's own (k, v) -- what `past` is made of."""
        recompute = self.checkpoint and self.training and torch.is_grad_enabled() and not want_kv and past is None
        kvs = []
        for i, layer in enumerate(self.layers):
            # Non-reentrant checkpointing keeps the RNG state, so dropout
            # masks in the recompute match the forward pass exactly.
            if spec is None:
                x = checkpoint(layer, x, use_reentrant=False) if recompute else layer(x)
            elif recompute:
                x = checkpoint(layer, x, spec, use_reentrant=False)
            else:
                x = layer(x, spec, None if past is None else past[i], want_kv)
                if want_kv:
                    x, kv = x
                    kvs.append(kv)
        return (x, kvs) if want_kv else x


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
