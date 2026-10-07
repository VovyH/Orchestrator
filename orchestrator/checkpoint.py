"""断点续训：checkpoint 的保存与恢复。

一个 checkpoint 目录里有什么：

    checkpoint-<step>/
    ├── trainer_state.json   # 步数、最优验证 loss、早停计数、rng 状态
    ├── model.safetensors    # 模型权重（LoRA 时是 adapter，体积很小）
    ├── optimizer.pt         # 优化器状态（AdamW 的 exp_avg / exp_avg_sq）
    └── scheduler.pt         # lr 调度器状态（当前退火到哪了）

恢复时四样全都要：只回权重不回优化器状态，动量会对不上，
lr 也会从预热重新开始——都等于换了一套超参继续训。
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import torch

logger = logging.getLogger("orchestrator")


def save_checkpoint(
    directory: str | Path,
    step: int,
    model,
    optimizer,
    scheduler,
    best_val_loss: float | None,
    patience_used: int,
) -> Path:
    """存一个 checkpoint-{step} 目录，返回其路径。"""
    root = Path(directory)
    target = root / f"checkpoint-{step}"
    target.mkdir(parents=True, exist_ok=True)

    # 模型。PeftModel.save_pretrained 只存 adapter（小）；HF 模型存全量；
    # TinyCausalLM 等裸 nn.Module 没有 save_pretrained，退回 torch.save
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(target, safe_serialization=True)
    else:
        torch.save(model.state_dict(), target / "model.pt")

    # 优化器和调度器。state_dict 是纯 tensor/dict，torch.save 直接可序列化
    torch.save(optimizer.state_dict(), target / "optimizer.pt")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), target / "scheduler.pt")

    # 训练进度 + 随机数状态：恢复后 shuffle / dropout 的序列才和中断前一致。
    # rng 状态用 torch.save 单独存 .pt（json 序列化不了 tensor/嵌套元组）
    state = {
        "step": step,
        "best_val_loss": best_val_loss,
        "patience_used": patience_used,
    }
    (target / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rng = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state()
    torch.save(rng, target / "rng_state.pt")
    logger.info("checkpoint 已存 %s", target)
    return target


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    """在输出目录里找步数最大的 checkpoint-*。没有则 None。"""
    root = Path(directory)
    if not root.is_dir():
        return None
    candidates = []
    for child in root.glob("checkpoint-*"):
        if child.is_dir() and (child / "trainer_state.json").is_file():
            try:
                step = int(child.name.split("-", 1)[1])
            except ValueError:
                continue
            candidates.append((step, child))
    return max(candidates)[1] if candidates else None


def load_checkpoint(
    target: str | Path,
    model,
    optimizer,
    scheduler,
) -> dict:
    """把 checkpoint 加载回 model / optimizer / scheduler，返回 trainer_state。"""
    target = Path(target)

    # 1. 模型权重。LoRA 时存的是 adapter，PeftModel.load_adapter 直接恢复；
    #    普通模型读 safetensors 后 loose 匹配。
    adapter = target / "adapter_model.safetensors"
    weights = target / "model.safetensors"
    pickle = target / "model.pt"
    if adapter.is_file():
        from peft import PeftModel

        if isinstance(model, PeftModel):
            model.load_adapter(str(target), "default", is_trainable=True)
            logger.info("LoRA adapter 已恢复")
    elif weights.is_file():
        from safetensors.torch import load_file

        state = load_file(str(weights))
        result = model.load_state_dict(state, strict=False)
        logger.info("模型权重恢复：missing=%d unexpected=%d", len(result.missing_keys), len(result.unexpected_keys))
    elif pickle.is_file():
        model.load_state_dict(torch.load(pickle, map_location="cpu", weights_only=True))
        logger.info("模型权重恢复（torch.save 格式）")

    # 2. 优化器
    if (target / "optimizer.pt").is_file():
        optimizer.load_state_dict(torch.load(target / "optimizer.pt", map_location="cpu", weights_only=True))
        logger.info("优化器状态已恢复")

    # 3. 调度器
    if scheduler is not None and (target / "scheduler.pt").is_file():
        scheduler.load_state_dict(torch.load(target / "scheduler.pt", map_location="cpu", weights_only=True))
        logger.info("lr 调度器状态已恢复")

    # 4. 进度 + rng
    state = json.loads((target / "trainer_state.json").read_text(encoding="utf-8"))
    if (target / "rng_state.pt").is_file():
        rng = torch.load(target / "rng_state.pt", weights_only=False)
        random.setstate(_tuple(rng["python"]))
        torch.set_rng_state(rng["torch"].to(dtype=torch.uint8))
        if "cuda" in rng and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state(rng["cuda"].to(dtype=torch.uint8))
            except Exception:
                pass  # 换了卡或 CPU 恢复时容忍失败，只影响 shuffle 复现
    return state


def _tuple(value):
    """python random.getstate 的嵌套元组，torch.load 恢复时 list 转回 tuple。"""
    if isinstance(value, list):
        return tuple(_tuple(item) for item in value)
    return value
