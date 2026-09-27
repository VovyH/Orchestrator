"""选择 GPU 或 PPU。

参数用 gpu / cuda，或 ppu。两边的计时、显存和探针走各自的运行时，
采样线程分别调用 nvidia-smi 和 ppu-smi。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.profiler import ProfilerActivity


@dataclass(frozen=True)
class Accelerator:
    kind: str
    device: torch.device
    smi: str | None
    activity: ProfilerActivity | None

    def synchronize(self) -> None:
        runtime = self._runtime()
        if runtime is not None:
            runtime.synchronize()

    def reset_peak(self) -> None:
        runtime = self._runtime()
        if runtime is not None and hasattr(runtime, "reset_peak_memory_stats"):
            runtime.reset_peak_memory_stats()

    def peak_mb(self) -> float | None:
        runtime = self._runtime()
        if runtime is None or not hasattr(runtime, "max_memory_allocated"):
            return None
        return runtime.max_memory_allocated() / (1024 * 1024)

    def empty_cache(self) -> None:
        runtime = self._runtime()
        if runtime is not None and hasattr(runtime, "empty_cache"):
            runtime.empty_cache()

    def event(self):
        runtime = self._runtime()
        if runtime is None or not hasattr(runtime, "Event"):
            return None
        return runtime.Event(enable_timing=True)

    def manual_seed_all(self, seed: int) -> None:
        runtime = self._runtime()
        if runtime is not None and hasattr(runtime, "manual_seed_all"):
            runtime.manual_seed_all(seed)

    def _runtime(self):
        if self.kind == "gpu":
            return torch.cuda
        if self.kind == "ppu":
            return getattr(torch, "ppu", None)
        return None


def canonical(name: str) -> str:
    kind = name.lower()
    if kind in {"gpu", "cuda"}:
        return "gpu"
    if kind in {"ppu", "cpu", "auto"}:
        return kind
    raise ValueError("device 只能是 auto、cpu、gpu、cuda 或 ppu")


def profiler_activity(kind: str) -> ProfilerActivity | None:
    if kind == "gpu":
        return ProfilerActivity.CUDA
    if kind != "ppu":
        return None
    for name in ("PPU", "PrivateUse1"):
        activity = getattr(ProfilerActivity, name, None)
        if activity is not None:
            return activity
    return None


def smi_binary(kind: str) -> str | None:
    if kind == "gpu":
        return "nvidia-smi"
    if kind == "ppu":
        return "ppu-smi"
    return None


def resolve_accelerator(name: str) -> Accelerator:
    kind = canonical(name)
    if kind == "auto":
        if torch.cuda.is_available():
            kind = "gpu"
        elif _ppu_ready():
            kind = "ppu"
        else:
            kind = "cpu"
    if kind == "cpu":
        return Accelerator("cpu", torch.device("cpu"), None, None)
    if kind == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("参数选了 gpu，但这台机器没有可用的 CUDA")
        return Accelerator("gpu", torch.device("cuda"), "nvidia-smi", ProfilerActivity.CUDA)
    if not _ppu_ready():
        raise RuntimeError("参数选了 ppu，但这台机器没有注册 torch 的 ppu 后端")
    return Accelerator("ppu", torch.device("ppu"), "ppu-smi", profiler_activity("ppu"))


def _ppu_ready() -> bool:
    if hasattr(torch, "ppu") and getattr(torch.ppu, "is_available", lambda: True)():
        return True
    getter = getattr(torch._C, "_get_privateuse1_backend_name", None)
    if getter is None:
        return False
    try:
        return getter() == "ppu"
    except Exception:
        return False
