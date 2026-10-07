"""侧车线程采样加速器。只调 nvidia-smi 或 ppu-smi，不在训练流上同步。"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

from .backend import canonical, smi_binary


@dataclass(frozen=True)
class GpuSample:
    timestamp: float
    util: float
    memory_mb: float
    total_mb: float


class GpuMonitor:
    def __init__(self, interval: float = 0.2, accelerator: str = "gpu"):
        self.interval = interval
        self.smi = smi_binary(canonical(accelerator))
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="device-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def window(self, start: float, end: float) -> list[GpuSample]:
        return [sample for sample in self.samples if start <= sample.timestamp <= end]

    def _loop(self) -> None:
        if self.smi is None:
            return
        while not self._stop.is_set():
            sample = _read_smi(self.smi)
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval)


def _read_smi(binary: str) -> GpuSample | None:
    try:
        completed = subprocess.run(
            [
                binary,
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    util, used, total = completed.stdout.strip().splitlines()[0].split(",")
    return GpuSample(time.time(), float(util), float(used), float(total))
