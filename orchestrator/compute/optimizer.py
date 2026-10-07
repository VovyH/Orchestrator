"""优化器：PPU 官方 FusedAdamW，其次 PyTorch fused=True，再退回普通 AdamW。"""

import logging

import torch

logger = logging.getLogger("orchestrator")


def create_optimizer(parameters, lr: float, betas: tuple[float, float], weight_decay: float, use_fused: bool = True):
    if not use_fused:
        logger.info("优化器：普通 AdamW（本次关闭 fused）")
        return torch.optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
    try:
        from ppu.optimizers import FusedAdamW

        logger.info("优化器：PPU FusedAdamW")
        return FusedAdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
    except ImportError:
        pass

    try:
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay, fused=True)
        logger.info("优化器：PyTorch AdamW fused=True")
        return optimizer
    except (TypeError, RuntimeError) as exc:
        logger.info("优化器：普通 AdamW（fused 不可用：%s）", exc)
        return torch.optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
