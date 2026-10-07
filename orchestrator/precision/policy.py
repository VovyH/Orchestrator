"""精度路由。

只有探测到硬件上的 FP8 矩阵乘才替换 Linear。探测失败就留在 BF16。
反向用 BF16 公式计算，避免软件模拟的 FP8 反向把速度打穿，同时保证梯度是对的。
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("orchestrator")

_FP8_MAX = 448.0


class _FP8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inputs, weight)
        return _fp8_matmul(inputs, weight)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, weight = ctx.saved_tensors
        grad_flat = grad_output.reshape(-1, weight.shape[0]).float()
        input_flat = inputs.reshape(-1, inputs.shape[-1]).float()
        grad_input = (grad_flat @ weight.float()).to(dtype=inputs.dtype).view_as(inputs)
        grad_weight = (grad_flat.transpose(0, 1) @ input_flat).to(dtype=weight.dtype)
        return grad_input, grad_weight


def _fp8_matmul(inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """inputs [..., K]，weight [N, K]，输出 [..., N]。K 和 N 须是 16 的倍数。"""
    flat = inputs.reshape(-1, inputs.shape[-1])
    if flat.shape[-1] % 16 != 0 or weight.shape[0] % 16 != 0:
        return torch.nn.functional.linear(inputs, weight)
    scale_input = (flat.detach().float().abs().max() / _FP8_MAX).clamp(min=1e-12)
    scale_weight = (weight.detach().float().abs().max() / _FP8_MAX).clamp(min=1e-12)
    left = (flat / scale_input).to(torch.float8_e4m3fn)
    right = (weight / scale_weight).to(torch.float8_e4m3fn).t()
    output = torch._scaled_mm(
        left,
        right,
        scale_a=scale_input.reshape(1),
        scale_b=scale_weight.reshape(1),
        out_dtype=torch.bfloat16,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.view(*inputs.shape[:-1], weight.shape[0]).to(dtype=inputs.dtype)


class FP8Linear(nn.Module):
    """主权重仍是 BF16 或 FP32，前向矩阵乘走 FP8 tensor core。"""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach())
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = _FP8LinearFn.apply(inputs, self.weight)
        if self.bias is not None:
            output = output + self.bias.to(dtype=output.dtype)
        return output


def probe_native_fp8(device: torch.device) -> tuple[bool, str]:
    if device.type not in {"cuda", "ppu"}:
        return False, "当前设备没有 CUDA/PPU"
    try:
        import importlib

        importlib.import_module("ppu.nn")
        return True, "检测到 ppu.nn"
    except ImportError:
        pass
    if not hasattr(torch, "_scaled_mm"):
        return False, "这版 PyTorch 没有 _scaled_mm"
    try:
        weight = torch.randn(32, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        inputs = torch.randn(32, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        output = _FP8LinearFn.apply(inputs, weight)
        output.sum().backward()
        if weight.grad is None or inputs.grad is None:
            return False, "FP8 前向没有接到反向"
        if not torch.isfinite(weight.grad).all():
            return False, "FP8 梯度不是有限数"
        return True, "torch._scaled_mm"
    except Exception as exc:
        return False, f"FP8 探测失败：{exc}"


def apply_precision(model: nn.Module, precision: str, device: torch.device) -> torch.dtype:
    requested = precision.lower()
    if device.type not in {"cuda", "ppu"}:
        logger.info("精度：FP32")
        return torch.float32

    if requested in ("fp8", "auto"):
        supported, reason = probe_native_fp8(device)
        if supported:
            _replace_linears(model)
            logger.info("精度：FP8（%s）。lm_head 保持原精度，交给分块 loss。", reason)
            return torch.bfloat16
        logger.info("精度：BF16。%s", reason)
        return torch.bfloat16

    logger.info("精度：BF16")
    return torch.bfloat16


def _replace_linears(model: nn.Module) -> None:
    for child_name, child in list(model.named_children()):
        if child_name == "lm_head":
            continue
        if isinstance(child, nn.Linear):
            setattr(model, child_name, FP8Linear(child))
        else:
            _replace_linears(child)
