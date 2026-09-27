"""隔若干层做一次 activation checkpoint，而不是每一层都重算。"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("orchestrator")


def layer_list(model: nn.Module) -> list[nn.Module] | None:
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
    ]
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(getattr(inner, "language_model", None))
    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return list(layers)
    return None


def apply_selective_checkpoint(model: nn.Module, every_n: int) -> None:
    if every_n <= 0:
        logger.info("checkpoint：关闭")
        return
    layers = layer_list(model)
    if not layers:
        logger.info("checkpoint：模型没有 layers，跳过")
        return
    wrapped = 0
    for index, layer in enumerate(layers):
        if index % every_n != 0:
            continue
        _wrap_forward(layer)
        wrapped += 1
    logger.info("checkpoint：每 %s 层重算一次，本轮包了 %s 层", every_n, wrapped)


def _wrap_forward(layer: nn.Module) -> None:
    original = layer.forward

    def forward(*args, **kwargs):
        if kwargs or not torch.is_grad_enabled():
            return original(*args, **kwargs)
        return torch.utils.checkpoint.checkpoint(original, *args, use_reentrant=False)

    layer.forward = forward
