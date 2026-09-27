"""训练配置。未知字段直接报错，避免 yaml 里写错了还静默忽略。"""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class TrainConfig:
    model_name_or_path: str = "tiny"
    dataset_path: str = ""
    max_samples: int = 0
    max_seq_len: int = 128
    target_tokens_per_step: int = 256
    synthetic_samples: int = 48
    synthetic_min_len: int = 8
    synthetic_max_len: int = 96
    vocab_size: int = 128
    tiny_hidden: int = 64
    tiny_layers: int = 2
    tiny_heads: int = 4
    precision: str = "bf16"  # bf16 | fp8 | auto。auto 在卡支持时走 FP8，否则留在 BF16
    fused_adam: bool = True
    packing: str = "auto"  # auto 试跑后决定；on 一直 packing；off 一直 batch
    nopack_batch_size: int = 2
    pack_waste_min: float = 0.10  # 并排时空隙低于这个比例，就不要 packing
    pack_length_limit: int = 4096  # 按分布算出的 max token 不超过这个数
    bucket_ratio: float = 1.25  # 桶内最长不超过最短的这个倍数
    packing_probe_steps: int = 0  # >0 时开训前各跑几步，第 1 步当预热
    compile_mlp: bool = False
    compile_attention: bool = False
    chunk_tokens: int = 128  # 词表分块。0 表示整段 logits 一次算出
    checkpoint_every_n: int = 0
    max_steps: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 42
    log_every: int = 1
    peak_tflops: float | None = None
    device: str = "auto"  # auto | cpu | gpu | cuda | ppu
    restrict_vocab: bool = False  # --crop-vocab：只留中文、英文和中英文符号


def load_config(path: str) -> TrainConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("配置文件的顶层必须是映射")
    known = {item.name for item in fields(TrainConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"配置里有无法识别的字段: {unknown}")
    if "betas" in raw and raw["betas"] is not None:
        raw["betas"] = tuple(raw["betas"])
    if "packing" in raw:
        raw["packing"] = _normalize_packing(raw["packing"])
    return TrainConfig(**raw)


def _normalize_packing(value):
    if value is True or (isinstance(value, str) and value.lower() in {"on", "true", "pack", "1"}):
        return "on"
    if value is False or (isinstance(value, str) and value.lower() in {"off", "false", "batch", "0"}):
        return "off"
    if isinstance(value, str) and value.lower() in {"auto", "adaptive"}:
        return "auto"
    raise ValueError(f"packing 只能是 auto、on 或 off，收到 {value!r}")
