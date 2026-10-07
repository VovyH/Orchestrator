"""一步的有效 token、padding 空隙、耗时和可选的 MFU。"""

from dataclasses import dataclass


@dataclass
class StepMeter:
    step: int
    loss: float
    content_tokens: int
    seconds: float
    padding_waste: float
    tokens_per_second: float
    mfu: float | None
    peak_memory_mb: float | None
    grad_norm: float | None = None
    gpu_util: float | None = None
    gpu_memory_mb: float | None = None

    def format(self) -> str:
        mfu = "-" if self.mfu is None else f"{self.mfu:.3f}"
        memory = "-" if self.peak_memory_mb is None else f"{self.peak_memory_mb:.0f}"
        grad = "-" if self.grad_norm is None else f"{self.grad_norm:.3f}"
        util = "-" if self.gpu_util is None else f"{self.gpu_util:.0f}"
        smi_mb = "-" if self.gpu_memory_mb is None else f"{self.gpu_memory_mb:.0f}"
        return (
            f"step={self.step} loss={self.loss:.4f} tokens={self.content_tokens} "
            f"waste={self.padding_waste:.4f} tok_s={self.tokens_per_second:.1f} "
            f"step_s={self.seconds:.3f} mfu={mfu} peak_mb={memory} "
            f"grad={grad} util={util}% smi_mb={smi_mb}"
        )


def padding_waste(content_tokens: int, cutoff_slots: int) -> float:
    if cutoff_slots <= 0:
        return 0.0
    return 1.0 - content_tokens / cutoff_slots


def model_flops_utilization(parameter_count: int, content_tokens: int, seconds: float, peak_tflops: float | None) -> float | None:
    if peak_tflops is None or peak_tflops <= 0 or seconds <= 0:
        return None
    flops = 6.0 * parameter_count * content_tokens
    return flops / (peak_tflops * 1e12 * seconds)
