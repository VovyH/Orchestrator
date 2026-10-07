"""一层耗时，以及这一层里面各个算子的自身耗时。

只包 transformer block，以及 block 里的 attn / mlp。
前向和反向各开一个 profiler 区间；区间内的 aten 算子按父区间归类。
探针本身会拖慢这一步，所以只在指定的 step 上打开。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

from .backend import canonical, profiler_activity


@dataclass
class _Bucket:
    inclusive_us: float = 0.0
    ops: dict[str, float] = field(default_factory=dict)


class _Attached:
    def __init__(self):
        self.handles: list = []
        self.restored: list[tuple[nn.Module, object]] = []

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        for module, original in self.restored:
            module.forward = original


def profile_step(model: nn.Module, step_fn, accelerator: str | None = None):
    """跑一步，返回 (step_fn 的结果, 文本报告)。accelerator 取 gpu/cuda 或 ppu。"""
    attached = _attach(model)
    kind = canonical(accelerator) if accelerator else _kind_of(model)
    activities = [ProfilerActivity.CPU]
    extra = profiler_activity(kind)
    if extra is not None:
        activities.append(extra)
    try:
        with profile(activities=activities, record_shapes=False) as prof:
            result = step_fn()
        _synchronize(model)
        report = _render(prof, kind)
    finally:
        attached.close()
    return result, report


def _kind_of(model: nn.Module) -> str:
    device_type = next(model.parameters()).device.type
    if device_type == "cuda":
        return "gpu"
    if device_type == "ppu":
        return "ppu"
    return "cpu"


def _synchronize(model: nn.Module) -> None:
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "ppu" and hasattr(torch, "ppu"):
        torch.ppu.synchronize()


def _attach(model: nn.Module) -> _Attached:
    attached = _Attached()
    for name, module in _blocks(model):
        _wrap_forward(attached, module, name)
        _hook_backward(attached, module, name)
    return attached


def _blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    found: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        if name.split(".")[-1] not in {"layers", "blocks"}:
            continue
        for index, block in enumerate(module):
            prefix = f"{name}.{index}" if name else str(index)
            found.append((prefix, block))
            for child_name, child in block.named_children():
                if child_name in {"attn", "self_attn", "mlp", "attention"}:
                    found.append((f"{prefix}.{child_name}", child))
    return found


def _wrap_forward(attached: _Attached, module: nn.Module, name: str) -> None:
    original = module.forward

    def forward(*args, **kwargs):
        with record_function(f"probe/{name}/forward"):
            return original(*args, **kwargs)

    module.forward = forward
    attached.restored.append((module, original))


def _hook_backward(attached: _Attached, module: nn.Module, name: str) -> None:
    def pre(mod, grad_output):
        ctx = record_function(f"probe/{name}/backward")
        ctx.__enter__()
        mod._probe_backward = ctx

    def post(mod, grad_input, grad_output):
        ctx = getattr(mod, "_probe_backward", None)
        if ctx is not None:
            ctx.__exit__(None, None, None)
            del mod._probe_backward

    attached.handles.append(module.register_full_backward_pre_hook(pre))
    attached.handles.append(module.register_full_backward_hook(post))


def _render(prof, kind: str) -> str:
    buckets: dict[str, _Bucket] = {}
    for event in prof.events():
        if event.name.startswith("probe/"):
            bucket = buckets.setdefault(event.name, _Bucket())
            bucket.inclusive_us += _inclusive_us(event)
            continue
        scope = _nearest_probe(event)
        if scope is None:
            continue
        self_us = _self_us(event)
        if self_us <= 0 or event.name.startswith("autograd::"):
            continue
        bucket = buckets.setdefault(scope, _Bucket())
        bucket.ops[event.name] = bucket.ops.get(event.name, 0.0) + self_us

    lines = [f"accelerator  {kind}", "layer  phase  ms  op  self_ms  share"]
    for name in sorted(buckets):
        bucket = buckets[name]
        layer, phase = _split_probe_name(name)
        inclusive_ms = bucket.inclusive_us / 1000.0
        lines.append(f"{layer}  {phase}  {inclusive_ms:.3f}")
        ranked = sorted(bucket.ops.items(), key=lambda item: item[1], reverse=True)
        shown = ranked[:8]
        hidden = ranked[8:]
        for op_name, self_us in shown:
            self_ms = self_us / 1000.0
            share = self_us / bucket.inclusive_us if bucket.inclusive_us else 0.0
            lines.append(f"  {op_name}  {self_ms:.3f}  {share:.0%}")
        if hidden:
            rest = sum(self_us for _, self_us in hidden)
            lines.append(f"  other  {rest / 1000.0:.3f}")
    if not buckets:
        lines.append("没有采到层。模型里需要名为 layers 或 blocks 的 ModuleList。")
    return "\n".join(lines) + "\n"


def _nearest_probe(event) -> str | None:
    node = event.cpu_parent
    while node is not None:
        if node.name.startswith("probe/"):
            return node.name
        node = node.cpu_parent
    return None


def _inclusive_us(event) -> float:
    if event.device_time_total:
        return float(event.device_time_total)
    return float(event.cpu_time_total)


def _self_us(event) -> float:
    if event.self_device_time_total:
        return float(event.self_device_time_total)
    return float(event.self_cpu_time_total)


def _split_probe_name(name: str) -> tuple[str, str]:
    body = name[len("probe/") :]
    layer, _, phase = body.rpartition("/")
    return layer, phase
