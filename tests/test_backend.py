"""gpu 和 ppu 只差在参数上：采样命令和 profiler 活动跟着变。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让同目录的 _bootstrap 可见
from _bootstrap import ensure_importable

ensure_importable()  # Orchestrator 是根项目还是子项目，都能 import orchestrator

from torch.profiler import ProfilerActivity

from orchestrator.backend import canonical, profiler_activity, resolve_accelerator, smi_binary


def test_gpu_and_cuda_share_one_backend():
    assert canonical("gpu") == canonical("cuda") == "gpu"
    assert smi_binary("gpu") == "nvidia-smi"
    assert profiler_activity("gpu") is ProfilerActivity.CUDA


def test_ppu_uses_its_own_smi_and_activity():
    assert canonical("ppu") == "ppu"
    assert smi_binary("ppu") == "ppu-smi"
    activity = profiler_activity("ppu")
    if hasattr(ProfilerActivity, "PPU"):
        expected = ProfilerActivity.PPU
    elif hasattr(ProfilerActivity, "PrivateUse1"):
        expected = ProfilerActivity.PrivateUse1
    else:
        expected = None
    assert activity is expected
    assert activity is not ProfilerActivity.CUDA


def test_cpu_has_no_smi():
    accelerator = resolve_accelerator("cpu")
    assert accelerator.kind == "cpu"
    assert accelerator.device.type == "cpu"
    assert accelerator.smi is None
    assert accelerator.event() is None


def test_missing_ppu_says_so():
    try:
        accelerator = resolve_accelerator("ppu")
    except RuntimeError as exc:
        assert "ppu" in str(exc)
        return
    assert accelerator.kind == "ppu"
    assert accelerator.smi == "ppu-smi"


def test_unknown_name_is_rejected():
    try:
        canonical("npu")
    except ValueError:
        return
    raise AssertionError("不认识的设备名应该报错")


if __name__ == "__main__":
    test_gpu_and_cuda_share_one_backend()
    test_ppu_uses_its_own_smi_and_activity()
    test_cpu_has_no_smi()
    test_missing_ppu_says_so()
    test_unknown_name_is_rejected()
    print("ok")
