"""小因果模型，用来在单机上验证 packing 和训练循环。

注意力只在 cu_seqlens 划出的子序列内部发生，位置编码由外部重置过的 position_ids 承担。
CUDA 上如果装了 flash_attn，长序列走 varlen，避免 T×T 的稠密 mask。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dense_causal_mask(cu_seqlens: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    total = int(cu_seqlens[-1])
    mask = torch.full((1, 1, total, total), torch.finfo(dtype).min, device=device, dtype=dtype)
    cuts = [int(item) for item in cu_seqlens.tolist()]
    for start, end in zip(cuts[:-1], cuts[1:]):
        span = end - start
        allowed = torch.ones(span, span, device=device, dtype=torch.bool).tril()
        mask[0, 0, start:end, start:end].masked_fill_(allowed, 0)
    return mask


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden 必须能被 heads 整除")
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        if cu_seqlens is not None and hidden.is_cuda and _flash_varlen_ready():
            mixed = self._flash(query, key, value, cu_seqlens)
        elif cu_seqlens is not None:
            mixed = self._dense(query, key, value, cu_seqlens)
        else:
            mixed = self._rows(query, key, value, padding_mask)
        return self.proj(mixed)

    def _dense(self, query, key, value, cu_seqlens) -> torch.Tensor:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) * (self.head_dim ** -0.5)
        scores = scores + dense_causal_mask(cu_seqlens, scores.dtype, scores.device)
        weights = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(weights, value)
        return mixed.transpose(1, 2).contiguous().view(mixed.shape[0], mixed.shape[2], -1)

    def _rows(self, query, key, value, padding_mask) -> torch.Tensor:
        """并排的 batch。每条自己做因果注意力，pad 的位置不参与。"""
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) * (self.head_dim ** -0.5)
        span = scores.shape[-1]
        causal = torch.ones(span, span, device=scores.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal.view(1, 1, span, span), torch.finfo(scores.dtype).min)
        if padding_mask is not None:
            valid = padding_mask.to(device=scores.device).bool()
            scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(weights, value)
        return mixed.transpose(1, 2).contiguous().view(mixed.shape[0], mixed.shape[2], -1)

    def _flash(self, query, key, value, cu_seqlens) -> torch.Tensor:
        from flash_attn import flash_attn_varlen_func

        cuts = [int(item) for item in cu_seqlens.tolist()]
        max_len = max(end - start for start, end in zip(cuts[:-1], cuts[1:]))
        out = flash_attn_varlen_func(
            query[0],
            key[0],
            value[0],
            cu_seqlens.to(device=query.device, dtype=torch.int32),
            cu_seqlens.to(device=query.device, dtype=torch.int32),
            max_len,
            max_len,
            causal=True,
        )
        return out.unsqueeze(0).contiguous().view(1, out.shape[0], -1)


class MLP(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.up = nn.Linear(hidden, hidden * 4, bias=False)
        self.down = nn.Linear(hidden * 4, hidden, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.up(hidden)))


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden)
        self.attn = Attention(hidden, heads)
        self.norm_mlp = nn.LayerNorm(hidden)
        self.mlp = MLP(hidden)

    def forward(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm_attn(hidden), cu_seqlens, padding_mask)
        hidden = hidden + self.mlp(self.norm_mlp(hidden))
        return hidden


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, hidden: int, layers: int, heads: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList(Block(hidden, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        hidden = self.embed(input_ids)
        if position_ids is not None:
            hidden = hidden + _position_table(self.embed.weight.shape[1], position_ids, hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, cu_seqlens, padding_mask)
        return self.norm(hidden)


def _position_table(hidden: int, position_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    # 固定的正弦表，只依赖已经按子序列重置过的 position_ids。
    device = position_ids.device
    half = hidden // 2
    freq = torch.arange(half, device=device, dtype=torch.float32)
    freq = 1.0 / (10000 ** (freq / max(half, 1)))
    angles = position_ids.float().unsqueeze(-1) * freq
    table = torch.zeros(*position_ids.shape, hidden, device=device, dtype=torch.float32)
    table[..., 0::2] = torch.sin(angles)
    if hidden % 2 == 0:
        table[..., 1::2] = torch.cos(angles)
    else:
        table[..., 1::2] = torch.cos(angles[..., : half])
    return table.to(dtype=dtype)


def _flash_varlen_ready() -> bool:
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError:
        return False
    return True
