"""HuggingFace 模型的注意力实现选择，以及 packing 时的 varlen flash。

拼长序列时不要再喂 [S,S] 的稠密 mask：那会把 sdpa 打回 math 核。
flash_attention_2 接 cu_seqlens 时，注意力只在每条子序列内部发生。
"""

from __future__ import annotations

import inspect
import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger("orchestrator")


def pick_attn_implementation() -> str:
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401

        logger.info("注意力：flash_attention_2（packing 走 varlen）")
        return "flash_attention_2"
    except ImportError:
        logger.info("注意力：sdpa（没装 flash_attn，packing 并排，不拼长序列）")
        return "sdpa"


def uses_flash(model) -> bool:
    configs = [getattr(model, "config", None)]
    root = configs[0]
    if root is not None:
        configs.append(getattr(root, "text_config", None))
        configs.append(getattr(root, "vision_config", None))
    inner = getattr(model, "model", None)
    if inner is not None:
        configs.append(getattr(inner, "config", None))
        language = getattr(inner, "language_model", None)
        if language is not None:
            configs.append(getattr(language, "config", None))
    for config in configs:
        if config is None:
            continue
        impl = getattr(config, "_attn_implementation", None) or getattr(config, "attn_implementation", "")
        if impl in {"flash_attention_2", "flash_attention_3"}:
            return True
    return False


def varlen_kwargs(cu_seqlens: torch.Tensor) -> dict:
    """传给 HF 的 FlashAttentionKwargs。cu_seqlens 必须是 int32。"""
    cu = cu_seqlens.to(dtype=torch.int32)
    if cu.numel() < 2:
        raise ValueError("cu_seqlens 至少要有起点和终点")
    max_len = int((cu[1:] - cu[:-1]).max().item())
    return {
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": max_len,
        "max_length_k": max_len,
        "is_causal": True,
    }


def _mask_modules(model) -> list:
    seen: set[int] = set()
    found = []
    stack = [model]
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        module = inspect.getmodule(type(obj))
        if module is not None and id(module) not in seen and hasattr(module, "create_causal_mask"):
            seen.add(id(module))
            found.append(module)
        for name in ("model", "language_model", "text_model"):
            child = getattr(obj, name, None)
            if child is not None and child is not obj:
                stack.append(child)
    return found


@contextmanager
def skip_causal_mask(model):
    """packing 时让 HF 不要再生成 4D causal mask，改由 cu_seqlens 隔离样本。"""
    patched = []
    for module in _mask_modules(model):
        patched.append((module, module.create_causal_mask))
        module.create_causal_mask = lambda *args, **kwargs: None
    try:
        yield
    finally:
        for module, original in patched:
            module.create_causal_mask = original
