"""单卡训练循环。

数据先装进窗口，再按 token 预算组成 step。loss 分块计算，优化器和精度按本机能力降级。
"""

import logging

import torch

from .backend import resolve_accelerator
from .compute.attention import pick_attn_implementation, skip_causal_mask, uses_flash, varlen_kwargs
from .compute.compile_blocks import compile_core_blocks
from .compute.optimizer import create_optimizer
from .config import TrainConfig
from .data.adaptive import PlannedStep, demote_packed, normalize_packing, plan_steps
from .data.collator import collate_plan
from .data.dataset import load_sequences
from .metrics import StepMeter, model_flops_utilization, padding_waste
from .modeling import TinyCausalLM
from .precision.checkpoint import apply_selective_checkpoint
from .precision.chunked_loss import chunked_causal_lm_loss
from .precision.policy import apply_precision
from .precision.vocab_softmax import crop_zh_en_vocab

logger = logging.getLogger("orchestrator")


def train(cfg: TrainConfig) -> list[StepMeter]:
    torch.manual_seed(cfg.seed)
    accelerator = resolve_accelerator(cfg.device)
    accelerator.manual_seed_all(cfg.seed)
    device = accelerator.device

    if cfg.model_name_or_path != "tiny" and not cfg.dataset_path:
        raise RuntimeError("HuggingFace 模型需要 dataset_path，每行一个 json：input_ids 或 text")
    sequences = load_sequences(cfg)
    lengths = [len(sequence) for sequence in sequences]
    mode = normalize_packing(cfg.packing)
    plans, dropped = _build_plans(lengths, cfg, mode)
    if not plans:
        raise RuntimeError("没有可训练的窗口：样本都长于 max_seq_len，或数据集是空的")

    model = _build_model(cfg, accelerator)
    model.to(device)
    if not isinstance(model, TinyCausalLM) and not uses_flash(model):
        plans = _demote_hf_without_flash(plans, lengths)
    dtype = apply_precision(model, cfg.precision, device)
    if dtype != next(model.parameters()).dtype:
        model.to(dtype=dtype)
    compile_core_blocks(model, cfg.compile_mlp, cfg.compile_attention)
    apply_selective_checkpoint(model, cfg.checkpoint_every_n)
    model.train()

    optimizer = create_optimizer(model.parameters(), cfg.lr, cfg.betas, cfg.weight_decay, cfg.fused_adam)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    restriction = None
    if cfg.restrict_vocab:
        if cfg.model_name_or_path == "tiny":
            raise RuntimeError("--crop-vocab 需要带 tokenizer 的模型，用来分辨中文、英文和符号")
        from transformers import AutoTokenizer

        vocab_size = _lm_weight(model).shape[0]
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
        restriction = crop_zh_en_vocab(tokenizer, vocab_size)
        logger.info(
            "词表裁剪：保留中文、英文和符号 %s / %s，分块 %s",
            restriction.size,
            vocab_size,
            cfg.chunk_tokens if cfg.chunk_tokens > 0 else "关闭",
        )
    else:
        logger.info("词表裁剪：关闭。分块 %s", cfg.chunk_tokens if cfg.chunk_tokens > 0 else "关闭")
    plans = _calibrate_packing(
        model, optimizer, sequences, plans, lengths, cfg, accelerator, parameter_count, restriction, mode
    )
    packed_steps = sum(step.packed for step in plans)
    logger.info(
        "样本 %s，丢弃 %s，步数 %s（packing %s，batch %s）",
        len(sequences),
        len(dropped),
        len(plans),
        packed_steps,
        len(plans) - packed_steps,
    )
    meters: list[StepMeter] = []

    for step_index in range(cfg.max_steps):
        plan = plans[step_index % len(plans)]
        batch = collate_plan(sequences, plan)
        meter = _run_step(model, optimizer, batch, cfg, accelerator, step_index + 1, parameter_count, restriction)
        meters.append(meter)
        if cfg.log_every > 0 and (step_index + 1) % cfg.log_every == 0:
            logger.info(meter.format())

    steady = meters[1:] or meters
    mean_tokens = sum(item.tokens_per_second for item in steady) / len(steady)
    mean_waste = sum(item.padding_waste for item in meters) / len(meters)
    logger.info("完成 %s 步，稳态 tok/s %.1f，平均 waste %.4f", len(meters), mean_tokens, mean_waste)
    return meters


def _run_step(model, optimizer, batch, cfg: TrainConfig, accelerator, step_number: int, parameter_count: int, restriction) -> StepMeter:
    device = accelerator.device
    accelerator.reset_peak()
    accelerator.synchronize()
    started = accelerator.event()
    ended = accelerator.event()
    clock = _cpu_clock()
    if started is not None:
        started.record()

    labels = batch.labels.to(device)
    hidden = _forward_hidden(model, batch, device)
    weight = _lm_weight(model)
    labels_for_loss = labels
    if restriction is not None:
        weight = restriction.project(weight)
        labels_for_loss = restriction.remap(labels)
    loss = chunked_causal_lm_loss(hidden, weight, labels_for_loss, cfg.chunk_tokens)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    if ended is not None:
        ended.record()
        accelerator.synchronize()
        seconds = started.elapsed_time(ended) / 1000.0
        peak_mb = accelerator.peak_mb()
    else:
        seconds = max(_cpu_clock() - clock, 1e-6)
        peak_mb = None

    waste = padding_waste(batch.content_tokens, batch.cutoff_slots)
    return StepMeter(
        step=step_number,
        loss=float(loss.detach().cpu()),
        content_tokens=batch.content_tokens,
        seconds=seconds,
        padding_waste=waste,
        tokens_per_second=batch.content_tokens / seconds,
        mfu=model_flops_utilization(parameter_count, batch.content_tokens, seconds, cfg.peak_tflops),
        peak_memory_mb=peak_mb,
    )


def _build_plans(lengths: list[int], cfg: TrainConfig, mode: str):
    layout = plan_steps(
        lengths,
        cfg.max_seq_len,
        mode,
        cfg.nopack_batch_size,
        cfg.pack_waste_min,
        cfg.pack_length_limit,
        cfg.bucket_ratio,
    )
    packed_steps = sum(step.packed for step in layout.steps)
    logger.info(
        "按长度分成 %s 个桶，max token %s（全数据 p90 的两倍，封顶 %s）。%s 步拼接走 varlen，%s 步并排",
        layout.buckets,
        layout.max_token,
        cfg.pack_length_limit,
        packed_steps,
        len(layout.steps) - packed_steps,
    )
    return layout.steps, layout.dropped


def _demote_hf_without_flash(plans: list[PlannedStep], lengths: list[int]) -> list[PlannedStep]:
    demoted: list[PlannedStep] = []
    changed = False
    for step in plans:
        if not step.packed:
            demoted.append(step)
            continue
        changed = True
        chosen = [lengths[index] for index in step.sample_ids]
        demoted.append(PlannedStep(step.sample_ids, False, step.content_tokens, len(chosen) * max(chosen)))
    if changed:
        logger.info("HuggingFace 没有 flash_attn，packing 步改成桶内并排")
    return demoted


def _calibrate_packing(model, optimizer, sequences, plans, lengths, cfg, accelerator, parameter_count, restriction, mode):
    """开训前用同一组样本各跑几步。batch 不慢于 packing，就把剩下的 packing 步拆回 batch。"""
    if mode != "auto" or cfg.packing_probe_steps <= 0:
        return plans
    packed = next((step for step in plans if step.packed), None)
    if packed is None:
        logger.info("分桶之后各步并排，块长按桶内长度算")
        return plans

    width = max(lengths[index] for index in packed.sample_ids)
    batched = PlannedStep(packed.sample_ids, False, packed.content_tokens, width * len(packed.sample_ids))
    weights = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    optimizer_state = optimizer.state_dict()
    pack_rate = _probe_rate(
        model, optimizer, sequences, packed, cfg, accelerator, parameter_count, restriction
    )
    _restore(model, optimizer, weights, optimizer_state)
    batch_rate = _probe_rate(
        model, optimizer, sequences, batched, cfg, accelerator, parameter_count, restriction
    )
    _restore(model, optimizer, weights, optimizer_state)
    logger.info("试跑 tok/s：packing %.0f，batch %.0f", pack_rate, batch_rate)
    if batch_rate >= pack_rate:
        logger.info("packing 没有更快，后面的步退回 batch")
        return demote_packed(plans, lengths, cfg.nopack_batch_size)
    logger.info("packing 更快，保留空隙够大的窗口")
    return plans


def _probe_rate(model, optimizer, sequences, step, cfg, accelerator, parameter_count, restriction) -> float:
    rate = 0.0
    for index in range(cfg.packing_probe_steps):
        batch = collate_plan(sequences, step)
        meter = _run_step(model, optimizer, batch, cfg, accelerator, index + 1, parameter_count, restriction)
        rate = meter.tokens_per_second
    return rate


def _restore(model, optimizer, weights, optimizer_state) -> None:
    model.load_state_dict(weights)
    optimizer.load_state_dict(optimizer_state)


def _forward_hidden(model, batch, device):
    input_ids = batch.input_ids.to(device)
    position_ids = batch.position_ids.to(device)
    if isinstance(model, TinyCausalLM):
        if batch.packed:
            return model(input_ids, batch.cu_seqlens.to(device), position_ids)
        return model(input_ids, None, position_ids, padding_mask=batch.padding_mask.to(device))
    inner = model.model if hasattr(model, "model") else model
    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "use_cache": False,
    }
    if not batch.packed:
        kwargs["attention_mask"] = batch.padding_mask.to(device)
        outputs = inner(**kwargs)
    elif uses_flash(model):
        kwargs.update(varlen_kwargs(batch.cu_seqlens.to(device)))
        with skip_causal_mask(model):
            outputs = inner(**kwargs, attention_mask=None)
    else:
        logger.info("没装 flash_attn，这一步改成并排，避免 [S,S] 稠密 mask")
        if batch.padding_mask is None:
            kwargs["attention_mask"] = torch.ones(input_ids.shape[:2], device=device, dtype=torch.long)
        else:
            kwargs["attention_mask"] = batch.padding_mask.to(device)
        outputs = inner(**kwargs)
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    return outputs[0]


def _lm_weight(model) -> torch.Tensor:
    if not hasattr(model, "lm_head"):
        raise TypeError("模型需要 lm_head，分块 loss 直接拿它的权重")
    return model.lm_head.weight


def _build_model(cfg: TrainConfig, accelerator) -> torch.nn.Module:
    if cfg.model_name_or_path == "tiny":
        return TinyCausalLM(cfg.vocab_size, cfg.tiny_hidden, cfg.tiny_layers, cfg.tiny_heads)
    from transformers import AutoModelForCausalLM

    dtype = torch.float32 if accelerator.kind == "cpu" else torch.bfloat16
    attn = pick_attn_implementation() if accelerator.kind != "cpu" else "sdpa"
    logger.info("加载 HuggingFace 模型 %s，注意力 %s", cfg.model_name_or_path, attn)
    return AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=attn,
        trust_remote_code=True,
    )


def _cpu_clock() -> float:
    import time

    return time.perf_counter()
