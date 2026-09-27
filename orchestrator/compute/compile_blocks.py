"""只编译 MLP 和 Attention 这两个块。全图编译在变长 packing 上容易 graph break。"""

import logging

import torch.nn as nn

from ..precision.checkpoint import layer_list

logger = logging.getLogger("orchestrator")


def compile_core_blocks(model: nn.Module, compile_mlp: bool, compile_attention: bool) -> None:
    if not compile_mlp and not compile_attention:
        logger.info("compile：关闭")
        return
    import torch

    layers = layer_list(model)
    if not layers:
        logger.info("compile：模型没有 layers，跳过")
        return
    compiled = 0
    for block in layers:
        if compile_mlp and hasattr(block, "mlp"):
            block.mlp = _compile(torch, block.mlp)
            compiled += 1
        if compile_attention:
            for name in ("attn", "self_attn", "linear_attn"):
                attention = getattr(block, name, None)
                if attention is None:
                    continue
                setattr(block, name, _compile(torch, attention))
                compiled += 1
    logger.info("compile：包了 %s 个模块（mlp=%s, attention=%s）", compiled, compile_mlp, compile_attention)


def _compile(torch, module: nn.Module) -> nn.Module:
    try:
        return torch.compile(module, mode="reduce-overhead")
    except Exception as exc:
        logger.info("compile 跳过一个模块：%s", exc)
        return module
