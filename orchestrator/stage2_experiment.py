"""Stage2 多模态消融。

同一批样本上比较四件事：packing、FP8、fused Adam、分块词表 loss。
restrict_vocab 由 --crop-vocab 打开：交叉熵只在中文、英文和中英文符号上重新 softmax。
不写 checkpoint。利用率由另一条线程采样，设备由 Ablation.device 选择 gpu 或 ppu。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .backend import resolve_accelerator
from .compute.attention import skip_causal_mask, uses_flash, varlen_kwargs
from .compute.optimizer import create_optimizer
from .data.adaptive import PlannedStep, plan_steps
from .monitor import GpuMonitor
from .precision.chunked_loss import chunked_causal_lm_loss
from .precision.policy import apply_precision
from .precision.vocab_softmax import crop_zh_en_vocab
from .probe import profile_step

logger = logging.getLogger("orchestrator")


@dataclass
class Ablation:
    name: str
    packing: bool
    precision: str
    fused_adam: bool
    chunk_tokens: int
    max_steps: int = 20
    max_seq_len: int = 8192
    nopack_batch_size: int = 2
    max_samples: int = 64
    lr: float = 1e-5
    seed: int = 42
    probe_step: int = 0
    device: str = "gpu"
    restrict_vocab: bool = False
    bucket_ratio: float = 1.25
    pack_length_limit: int = 4096


@dataclass
class EncodedSample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    mm_token_type_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    position_ids: torch.Tensor


def run_ablation(ablation: Ablation, samples: list[EncodedSample], model, output_dir: Path, tokenizer=None) -> dict:
    torch.manual_seed(ablation.seed)
    accelerator = resolve_accelerator(ablation.device)
    accelerator.manual_seed_all(ablation.seed)
    device = accelerator.device
    model.to(device)
    dtype = apply_precision(model, ablation.precision, device)
    if next(model.parameters()).dtype != dtype:
        model.to(dtype=dtype)
    model.train()
    optimizer = create_optimizer(model.parameters(), ablation.lr, (0.9, 0.95), 0.0, ablation.fused_adam)
    steps = _build_steps(samples, ablation)
    if ablation.packing and not uses_flash(model):
        steps = _demote_to_batch(steps, samples)
        logger.info("没装 flash_attn，packing 改成桶内并排，避免 [S,S] 稠密 mask")
    if ablation.packing and ablation.chunk_tokens <= 0:
        logger.info("packing 步会自动打开分块 loss，避免整段 logits 把长块拖慢")
    if len(steps) < ablation.max_steps:
        raise RuntimeError(f"{ablation.name} 只有 {len(steps)} 个 step，不够 {ablation.max_steps}")
    restriction = None
    if ablation.restrict_vocab:
        if tokenizer is None:
            raise RuntimeError("词表裁剪需要 tokenizer，请用 --crop-vocab 启动")
        vocab_size = model.lm_head.weight.shape[0]
        restriction = crop_zh_en_vocab(tokenizer, vocab_size)
        logger.info("词表裁剪：保留中文、英文和符号 %s / %s", restriction.size, vocab_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = GpuMonitor(interval=0.2, accelerator=accelerator.kind)
    monitor.start()
    records = []
    run_started = time.time()
    try:
        for index in range(ablation.max_steps):
            group = [samples[sample_id] for sample_id in steps[index].sample_ids]
            batch = collate_samples(group, device, dtype, steps[index].packed)
            if ablation.probe_step == index + 1:
                record, report = profile_step(
                    model,
                    lambda: _train_step(model, optimizer, batch, ablation, monitor, index + 1, accelerator, restriction),
                    accelerator=accelerator.kind,
                )
                (output_dir / "probe.txt").write_text(report, encoding="utf-8")
                logger.info("探针（这一步的耗时含探针开销）\n%s", report)
            else:
                record = _train_step(model, optimizer, batch, ablation, monitor, index + 1, accelerator, restriction)
            records.append(record)
            logger.info(
                "%s step=%s loss=%.4f tokens=%s waste=%.4f prefill=%.3fs backward=%.3fs step=%.3fs util=%.0f mem=%s",
                ablation.name,
                record["step"],
                record["loss"],
                record["tokens"],
                record["padding_waste"],
                record["prefill_s"],
                record["backward_s"],
                record["step_s"],
                record["gpu_util_mean"],
                record["mem_max_mb"],
            )
    finally:
        monitor.stop()
    summary = _summarize(ablation, records, monitor, run_started, time.time(), None if restriction is None else restriction.size)
    if restriction is not None:
        torch.save(restriction.kept_ids, output_dir / "kept_token_ids.pt")
    (output_dir / "steps.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def encode_stage2(model_path: str, json_path: str, image_root: str, max_samples: int, model) -> list[EncodedSample]:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    rows = json.loads(Path(json_path).read_text(encoding="utf-8"))
    encoded: list[EncodedSample] = []
    for row in rows:
        if len(encoded) >= max_samples:
            break
        image_path = Path(image_root) / Path(row["image"]).name
        if not image_path.exists():
            continue
        sample = _encode_row(processor, model, row, image_path)
        if sample is not None:
            encoded.append(sample)
    if len(encoded) < 8:
        raise RuntimeError(f"可用 Stage2 样本只有 {len(encoded)}")
    logger.info("编码 Stage2 样本 %s，长度 %s-%s", len(encoded), min(item.input_ids.numel() for item in encoded), max(item.input_ids.numel() for item in encoded))
    return encoded


def _encode_row(processor, model, row: dict, image_path: Path) -> EncodedSample | None:
    user = str(row["conversations"][0]["content"]).replace("<image>", "").strip()
    assistant = str(row["conversations"][1]["content"]).strip()
    if not user or not assistant:
        return None
    image = Image.open(image_path).convert("RGB")
    full_text = processor.apply_chat_template(_messages(user, assistant, image_path), tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(_messages(user, None, image_path), tokenize=False, add_generation_prompt=True)
    full = processor(text=[full_text], images=[image], return_tensors="pt")
    prompt = processor(text=[prompt_text], images=[image], return_tensors="pt")
    input_ids = full["input_ids"][0]
    prompt_ids = prompt["input_ids"][0]
    prompt_len = int(prompt_ids.numel())
    if prompt_len >= input_ids.numel() or not torch.equal(input_ids[:prompt_len], prompt_ids):
        return None
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    position_ids = _rope_positions(model, full)
    return EncodedSample(
        input_ids=input_ids.cpu(),
        labels=labels.cpu(),
        mm_token_type_ids=full["mm_token_type_ids"][0].cpu(),
        pixel_values=full["pixel_values"].cpu(),
        image_grid_thw=full["image_grid_thw"].cpu(),
        position_ids=position_ids.cpu(),
    )


def _messages(user: str, assistant: str | None, image_path: Path) -> list[dict]:
    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": user},
        ],
    }
    if assistant is None:
        return [user_message]
    return [user_message, {"role": "assistant", "content": [{"type": "text", "text": assistant}]}]


def _rope_positions(model, batch) -> torch.Tensor:
    inner = model.model if hasattr(model, "model") else model
    position_ids, _ = inner.get_rope_index(
        input_ids=batch["input_ids"],
        mm_token_type_ids=batch["mm_token_type_ids"],
        image_grid_thw=batch["image_grid_thw"],
        attention_mask=batch["attention_mask"],
    )
    if position_ids.dim() == 3:
        position_ids = position_ids[:, 0, :]
    return position_ids


def _build_steps(samples: list[EncodedSample], ablation: Ablation) -> list[PlannedStep]:
    lengths = [int(sample.input_ids.numel()) for sample in samples]
    if not ablation.packing:
        groups: list[PlannedStep] = []
        for start in range(0, len(samples), ablation.nopack_batch_size):
            group = tuple(range(start, min(start + ablation.nopack_batch_size, len(samples))))
            if len(group) != ablation.nopack_batch_size:
                continue
            chosen = [lengths[index] for index in group]
            groups.append(PlannedStep(group, False, sum(chosen), len(chosen) * max(chosen)))
        return groups
    layout = plan_steps(
        lengths,
        ablation.max_seq_len,
        "on",
        ablation.nopack_batch_size,
        0.10,
        ablation.pack_length_limit,
        ablation.bucket_ratio,
    )
    logger.info(
        "packing 按长度分成 %s 个桶，max token %s，%s 步（桶内拼接，varlen flash）",
        layout.buckets,
        layout.max_token,
        len(layout.steps),
    )
    return layout.steps


def _demote_to_batch(steps: list[PlannedStep], samples: list[EncodedSample]) -> list[PlannedStep]:
    lengths = [int(sample.input_ids.numel()) for sample in samples]
    demoted: list[PlannedStep] = []
    for step in steps:
        if not step.packed:
            demoted.append(step)
            continue
        chosen = [lengths[index] for index in step.sample_ids]
        demoted.append(PlannedStep(step.sample_ids, False, step.content_tokens, len(chosen) * max(chosen)))
    return demoted


def collate_samples(samples: list[EncodedSample], device, dtype, packing: bool) -> dict:
    if packing:
        return _collate_packed(samples, device, dtype)
    return _collate_padded(samples, device, dtype)


def _collate_packed(samples: list[EncodedSample], device, dtype) -> dict:
    lengths = [int(sample.input_ids.numel()) for sample in samples]
    cuts = [0]
    for length in lengths:
        cuts.append(cuts[-1] + length)
    input_ids = torch.cat([sample.input_ids for sample in samples]).unsqueeze(0).to(device)
    labels = torch.cat([sample.labels for sample in samples]).unsqueeze(0).to(device)
    for boundary in cuts[1:-1]:
        labels[0, boundary] = -100
    mm = torch.cat([sample.mm_token_type_ids for sample in samples]).unsqueeze(0).to(device)
    position_ids = torch.cat([sample.position_ids for sample in samples], dim=-1).unsqueeze(1).to(device)
    total = cuts[-1]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "mm_token_type_ids": mm,
        "position_ids": position_ids,
        "pixel_values": torch.cat([sample.pixel_values for sample in samples], dim=0).to(device=device, dtype=dtype),
        "image_grid_thw": torch.cat([sample.image_grid_thw for sample in samples], dim=0).to(device),
        "attention_mask": torch.ones(1, total, device=device, dtype=torch.long),
        "cu_seqlens": torch.tensor(cuts, dtype=torch.int32, device=device),
        "packed": True,
        "tokens": int(sum(lengths)),
        "slots": int(sum(lengths)),
        "samples": len(samples),
    }


def _collate_padded(samples: list[EncodedSample], device, dtype) -> dict:
    width = max(int(sample.input_ids.numel()) for sample in samples)
    ids, labels, mm, mask = [], [], [], []
    positions = []
    content = 0
    for sample in samples:
        length = int(sample.input_ids.numel())
        content += length
        pad = width - length
        ids.append(F.pad(sample.input_ids, (0, pad)))
        labels.append(F.pad(sample.labels, (0, pad), value=-100))
        mm.append(F.pad(sample.mm_token_type_ids, (0, pad)))
        mask.append(torch.cat([torch.ones(length), torch.zeros(pad)]))
        positions.append(F.pad(sample.position_ids, (0, pad)))
    return {
        "input_ids": torch.stack(ids).to(device),
        "labels": torch.stack(labels).to(device),
        "mm_token_type_ids": torch.stack(mm).to(device),
        "position_ids": torch.stack(positions, dim=1).to(device),
        "pixel_values": torch.cat([sample.pixel_values for sample in samples], dim=0).to(device=device, dtype=dtype),
        "image_grid_thw": torch.cat([sample.image_grid_thw for sample in samples], dim=0).to(device),
        "attention_mask": torch.stack(mask).to(device=device, dtype=torch.long),
        "packed": False,
        "tokens": content,
        "slots": width * len(samples),
        "samples": len(samples),
    }


def _train_step(model, optimizer, batch, ablation: Ablation, monitor: GpuMonitor, step: int, accelerator, restriction) -> dict:
    accelerator.reset_peak()
    start = accelerator.event()
    prefill_end = accelerator.event()
    end = accelerator.event()
    wall_start = time.time()
    if start is not None:
        start.record()
    hidden = _forward_hidden(model, batch)
    chunk_tokens = ablation.chunk_tokens
    if batch.get("packed") and chunk_tokens <= 0:
        chunk_tokens = 2048
    loss = _loss(hidden, model.lm_head.weight, batch["labels"], chunk_tokens, restriction)
    if prefill_end is not None:
        prefill_end.record()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if end is not None:
        end.record()
        accelerator.synchronize()
    wall_end = time.time()
    peak_mb = accelerator.peak_mb()
    if start is not None:
        prefill_s = start.elapsed_time(prefill_end) / 1000.0
        step_s = start.elapsed_time(end) / 1000.0
    else:
        step_s = max(wall_end - wall_start, 1e-6)
        prefill_s = step_s
    gpu_rows = monitor.window(wall_start, wall_end)
    utils = [row.util for row in gpu_rows]
    mems = [row.memory_mb for row in gpu_rows]
    return {
        "step": step,
        "loss": float(loss.detach().cpu()),
        "tokens": batch["tokens"],
        "slots": batch["slots"],
        "samples": batch["samples"],
        "padding_waste": 1.0 - batch["tokens"] / batch["slots"],
        "prefill_s": prefill_s,
        "backward_s": step_s - prefill_s,
        "step_s": step_s,
        "gpu_util_mean": sum(utils) / len(utils) if utils else -1,
        "gpu_util_max": max(utils) if utils else -1,
        "mem_max_mb": max(mems) if mems else -1,
        "torch_peak_mb": peak_mb if peak_mb is not None else -1,
        "monitor_samples": len(gpu_rows),
    }


def _forward_hidden(model, batch):
    inner = model.model if hasattr(model, "model") else model
    kwargs = {
        "input_ids": batch["input_ids"],
        "position_ids": batch["position_ids"],
        "pixel_values": batch["pixel_values"],
        "image_grid_thw": batch["image_grid_thw"],
        "mm_token_type_ids": batch["mm_token_type_ids"],
        "use_cache": False,
    }
    if batch.get("packed") and batch.get("cu_seqlens") is not None and uses_flash(model):
        kwargs.update(varlen_kwargs(batch["cu_seqlens"]))
        with skip_causal_mask(model):
            outputs = inner(**kwargs, attention_mask=None)
    else:
        outputs = inner(**kwargs, attention_mask=batch["attention_mask"])
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    return outputs[0]


def _loss(hidden, weight, labels, chunk_tokens: int, restriction=None) -> torch.Tensor:
    if restriction is not None:
        weight = restriction.project(weight)
        labels = restriction.remap(labels)
    if chunk_tokens > 0:
        return chunked_causal_lm_loss(hidden, weight, labels, chunk_tokens)
    shift_hidden = hidden[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    logits = F.linear(shift_hidden, weight)
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), shift_labels.view(-1), ignore_index=-100)


def _summarize(ablation: Ablation, records: list[dict], monitor: GpuMonitor, started: float, ended: float, vocab_kept: int | None) -> dict:
    steady = records[1:] or records
    rows = monitor.window(started, ended)
    utils = [row.util for row in rows]
    mems = [row.memory_mb for row in rows]

    def mean(key: str) -> float:
        return sum(item[key] for item in steady) / len(steady)

    return {
        "name": ablation.name,
        "packing": ablation.packing,
        "precision": ablation.precision,
        "fused_adam": ablation.fused_adam,
        "chunk_tokens": ablation.chunk_tokens,
        "restrict_vocab": ablation.restrict_vocab,
        "vocab_kept": vocab_kept if vocab_kept is not None else -1,
        "steps": len(records),
        "prefill_s_step1": records[0]["prefill_s"],
        "step_s_step1": records[0]["step_s"],
        "loss_first": records[0]["loss"],
        "loss_last": records[-1]["loss"],
        "steady_prefill_s": mean("prefill_s"),
        "steady_backward_s": mean("backward_s"),
        "steady_step_s": mean("step_s"),
        "steady_tokens": mean("tokens"),
        "steady_tok_s": mean("tokens") / mean("step_s"),
        "steady_waste": mean("padding_waste"),
        "steady_samples": mean("samples"),
        "steady_torch_peak_mb": mean("torch_peak_mb"),
        "monitor_util_mean": sum(utils) / len(utils) if utils else -1,
        "monitor_util_max": max(utils) if utils else -1,
        "monitor_mem_max_mb": max(mems) if mems else -1,
        "settings": asdict(ablation),
    }
