"""单卡训练循环。

数据先装进窗口，再按 token 预算组成 step。loss 分块计算，优化器和精度按本机能力降级。
"""

import logging
from dataclasses import dataclass

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


@dataclass
class TrainResult:
    """一次训练的结果：每步指标、训完的模型，以及调度器给出的步数。"""

    meters: list[StepMeter]
    model: torch.nn.Module
    steps_per_epoch: int
    samples: int
    dropped: int

    @property
    def final_loss(self) -> float:
        return self.meters[-1].loss if self.meters else float("nan")

    @property
    def tokens_per_second(self) -> float:
        steady = self.meters[1:] or self.meters
        if not steady:
            return 0.0
        return sum(item.tokens_per_second for item in steady) / len(steady)

    def summary(self) -> dict:
        peaks = [item.peak_memory_mb for item in self.meters if item.peak_memory_mb]
        return {
            "steps": len(self.meters),
            "steps_per_epoch": self.steps_per_epoch,
            "samples": self.samples,
            "dropped_samples": self.dropped,
            "final_loss": self.final_loss,
            "tokens_per_second": round(self.tokens_per_second, 1),
            "peak_memory_mb": round(max(peaks)) if peaks else None,
        }


def train(cfg: TrainConfig) -> list[StepMeter]:
    """老用法：只给配置，返回每一步的指标。"""
    return train_model(cfg).meters


def train_model(
    cfg: TrainConfig,
    sequences: list[list[int]] | None = None,
    tokenizer=None,
) -> TrainResult:
    """训练并返回模型。sequences 可以由调用方预先分好词传进来。"""
    torch.manual_seed(cfg.seed)
    accelerator = resolve_accelerator(cfg.device)
    accelerator.manual_seed_all(cfg.seed)
    device = accelerator.device

    if sequences is None and cfg.model_name_or_path != "tiny" and not cfg.dataset_path:
        raise RuntimeError("HuggingFace 模型需要 dataset_path，每行一个 json：input_ids 或 text")
    if sequences is None:
        sequences = load_sequences(cfg, tokenizer)
    lengths = [len(sequence) for sequence in sequences]
    mode = normalize_packing(cfg.packing)
    plans, dropped = _build_plans(lengths, cfg, mode)
    if not plans:
        raise RuntimeError("没有可训练的窗口：样本都长于 max_seq_len，或数据集是空的")

    model = _build_model(cfg, accelerator)
    model.to(device)
    if not isinstance(model, TinyCausalLM) and not _supports_varlen(model):
        plans = _demote_hf_without_flash(plans, lengths)
    model = _apply_lora(model, cfg)
    dtype = apply_precision(model, cfg.precision, device)
    if dtype != next(model.parameters()).dtype:
        model.to(dtype=dtype)
    compile_core_blocks(model, cfg.compile_mlp, cfg.compile_attention)
    apply_selective_checkpoint(model, cfg.checkpoint_every_n)
    model.train()

    if cfg.lora_r > 0:
        trainable = [p for p in model.parameters() if p.requires_grad]
        logger.info("LoRA：可训练参数 %.2fM / %.2fM", sum(p.numel() for p in trainable) / 1e6,
                    sum(p.numel() for p in model.parameters()) / 1e6)
        optimizer = create_optimizer(trainable, cfg.lr, cfg.betas, cfg.weight_decay, cfg.fused_adam)
    else:
        optimizer = create_optimizer(model.parameters(), cfg.lr, cfg.betas, cfg.weight_decay, cfg.fused_adam)
    scheduler = _build_scheduler(optimizer, cfg)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    restriction = None
    if cfg.restrict_vocab:
        if cfg.model_name_or_path == "tiny":
            raise RuntimeError("--crop-vocab 需要带 tokenizer 的模型，用来分辨中文、英文和符号")
        from transformers import AutoTokenizer

        vocab_size = _lm_weight(model).shape[0]
        if tokenizer is None:
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
    monitor = _start_monitor(accelerator)
    meters: list[StepMeter] = []
    bar = _make_progress(cfg)
    cursor = [0]  # monitor.samples 已消费到哪，配合步长 < 采样间隔的情况

    # ---- shuffle：打乱步的顺序，消除「分桶按长度排队喂」的偏置 ----
    # 注意 plans 是排序产物（短→长），不打乱时模型先学简单再被难题连续轰炸，
    # loss 曲线会呈假性爬坡（不是过拟合，是数据顺序偏置）。
    if cfg.shuffle:
        import random as _random

        _random.Random(cfg.seed).shuffle(plans)
        logger.info("步顺序已按 seed=%s 打乱", cfg.seed)

    # ---- 断点续训：输出目录里有 checkpoint 就恢复 ----
    start_step = 0
    best_val_loss: float | None = None
    patience_used = 0
    if cfg.resume_from:
        from .checkpoint import find_latest_checkpoint, load_checkpoint

        latest = find_latest_checkpoint(cfg.resume_from)
        if latest is not None:
            state = load_checkpoint(latest, model, optimizer, scheduler)
            start_step = int(state["step"])
            best_val_loss = state.get("best_val_loss")
            patience_used = int(state.get("patience_used") or 0)
            if start_step >= cfg.max_steps:
                logger.info("checkpoint 已到 %s 步（>= max_steps %s），本次不训，直接返回当前模型",
                            start_step, cfg.max_steps)
            else:
                logger.info("从 %s 恢复：已完成 %s 步，继续训到 %s 步", latest, start_step, cfg.max_steps)
        else:
            logger.info("resume_from=%s 里没有 checkpoint，从头开始", cfg.resume_from)

    # ---- 早停：验证集 loss 连续 patience 次不创新低就停 ----
    # 验证集必须和训练集走同一条编码路径（fit 用 encode_rows，认 text/input_ids/messages），
    # 不能用 load_sequences——它只认 input_ids/text，messages 格式的验证集会直接报错。
    val_sequences = None
    if cfg.early_stop_patience > 0 and cfg.val_dataset_path:
        from .api import encode_rows, _read_rows

        limit = min(cfg.max_samples or 10**9, 512)  # 验证集上限 512 条，防止验证耗时过长
        val_sequences, _, _ = encode_rows(
            _read_rows(cfg.val_dataset_path),
            tokenizer,
            max_seq_len=cfg.max_seq_len,
            max_samples=limit,
            truncate_long=True,
        )
        if not val_sequences:
            val_sequences = None
            logger.info("验证集为空，早停关闭")
        else:
            logger.info("早停：验证集 %s 条，每 %s 步检查一次，patience=%s",
                        len(val_sequences), cfg.early_stop_every, cfg.early_stop_patience)

    try:
        for step_index in range(start_step, cfg.max_steps):
            plan = plans[step_index % len(plans)]
            batch = collate_plan(sequences, plan)
            step_started = _wall_clock()
            meter = _run_step(model, optimizer, batch, cfg, accelerator, step_index + 1, parameter_count, restriction, scheduler)
            _attach_gpu_stats(monitor, meter, step_started, cursor)
            meters.append(meter)
            if bar is not None:
                bar.update(1)
                bar.set_postfix(
                    loss=f"{meter.loss:.3f}",
                    tok_s=f"{meter.tokens_per_second:.0f}",
                    grad="-" if meter.grad_norm is None else f"{meter.grad_norm:.2f}",
                    util="-" if meter.gpu_util is None else f"{meter.gpu_util:.0f}%",
                )
                if cfg.log_every > 0 and (step_index + 1) % cfg.log_every == 0:
                    bar.write(meter.format())
            elif cfg.log_every > 0 and (step_index + 1) % cfg.log_every == 0:
                logger.info(meter.format())

            # 定期存 checkpoint（需要 resume_from 作为存放目录——fit 场景即输出目录）
            if cfg.save_every > 0 and cfg.resume_from and (step_index + 1) % cfg.save_every == 0 and (step_index + 1) < cfg.max_steps:
                from .checkpoint import save_checkpoint

                save_checkpoint(cfg.resume_from, step_index + 1, model, optimizer, scheduler,
                                best_val_loss, patience_used)

            # 早停检查
            if val_sequences is not None and (step_index + 1) % cfg.early_stop_every == 0:
                val_loss = _validation_loss(model, val_sequences, cfg, accelerator, restriction)
                improved = best_val_loss is None or val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    patience_used = 0
                    logger.info("验证 loss 新低 %.4f（step %s）", val_loss, step_index + 1)
                else:
                    patience_used += 1
                    logger.info("验证 loss %.4f 未超过最优 %.4f（patience %s/%s）",
                                val_loss, best_val_loss, patience_used, cfg.early_stop_patience)
                if patience_used >= cfg.early_stop_patience:
                    logger.info("早停：连续 %s 次验证 loss 无改善，停在 step %s",
                                cfg.early_stop_patience, step_index + 1)
                    break
    finally:
        if bar is not None:
            bar.close()
        if monitor is not None:
            monitor.stop()

    steady = meters[1:] or meters
    mean_tokens = sum(item.tokens_per_second for item in steady) / len(steady)
    mean_waste = sum(item.padding_waste for item in meters) / len(meters)
    logger.info("完成 %s 步，稳态 tok/s %.1f，平均 waste %.4f", len(meters), mean_tokens, mean_waste)
    utils = [item.gpu_util for item in meters if item.gpu_util is not None]
    if utils:
        logger.info("GPU 利用率均值 %.0f%%，smi 显存均值 %.0f MiB",
                    sum(utils) / len(utils),
                    sum(item.gpu_memory_mb for item in meters if item.gpu_memory_mb) / len(utils))
    return TrainResult(
        meters=meters,
        model=model,
        steps_per_epoch=len(plans),
        samples=len(sequences),
        dropped=len(dropped),
    )


def _start_monitor(accelerator):
    """GPU/PPU 上开侧车线程采 nvidia-smi / ppu-smi；CPU 返回 None。"""
    if accelerator.kind not in {"gpu", "ppu"}:
        return None
    from .monitor import GpuMonitor

    monitor = GpuMonitor(interval=0.1, accelerator=accelerator.kind)
    monitor.start()
    return monitor


def _attach_gpu_stats(monitor, meter: StepMeter, started: float, cursor: list[int]) -> None:
    """把 smi 采样均值挂到 meter 上。

    步长常常小于采样间隔，一步的窗口里可能一个样本都没有。
    优先用本步窗口；没有就用自上一步以来新到的样本；再没有就退回最近一条。
    """
    if monitor is None:
        return
    samples = monitor.window(started, _wall_clock())
    if not samples:
        fresh = monitor.samples[cursor[0] :]
        if fresh:
            samples = fresh
        elif monitor.samples:
            samples = monitor.samples[-1:]
    cursor[0] = len(monitor.samples)
    if not samples:
        return
    meter.gpu_util = sum(sample.util for sample in samples) / len(samples)
    meter.gpu_memory_mb = sum(sample.memory_mb for sample in samples) / len(samples)


def _make_progress(cfg: TrainConfig):
    """有 tqdm 且开了 progress_bar 才建进度条；否则 None，走原来的 logger 行。"""
    if not cfg.progress_bar:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        logger.info("进度条：没装 tqdm，退回逐行日志（pip install tqdm 可开）")
        return None
    return tqdm(total=cfg.max_steps, unit="step", dynamic_ncols=True)


def _wall_clock() -> float:
    import time

    return time.time()


def _run_step(model, optimizer, batch, cfg: TrainConfig, accelerator, step_number: int, parameter_count: int, restriction, scheduler=None) -> StepMeter:
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
    clip_max = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max)
    try:
        grad_norm = float(total_norm)
    except (TypeError, RuntimeError):
        grad_norm = None
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

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
        grad_norm=grad_norm,
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


def _supports_varlen(model) -> bool:
    """模型能不能吃 varlen packing 的 FlashAttentionKwargs（cu_seq_lens_q 等）。

    装了 flash_attn 还不够：Qwen2 等纯文本模型的 decoder.forward 带
    **flash_attn_kwargs，Qwen2.5-VL 这类多模态模型的 forward 没有这些参数，
    传进去直接 TypeError。所以两关都过才走拼接：注意力实现是 flash，
    且 decoder.forward 签名收 varlen 关键字。注意要检查的是 decoder
    （经 _decoder_of 剥掉 PeftModel 外层），不是带 lm_head 的完整模型——
    LoraModel.forward 自带 **kwargs，会误判所有 LoRA 模型都支持。
    """
    if not uses_flash(model):
        return False
    import inspect

    inner = _decoder_of(model)
    try:
        parameters = inspect.signature(inner.forward).parameters
    except (ValueError, TypeError):
        return False
    if "cu_seq_lens_q" in parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


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
        logger.info("模型不支持 varlen packing（没装 flash_attn，或 forward 不收 FlashAttentionKwargs），拼接步改成桶内并排")
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


def _decoder_of(model):
    """拿到真正产出 hidden 的 decoder（Qwen2Model / Qwen2_5_VLModel 等）。

    PeftModel 外面包着 LoraModel：model.model 拿到的是带 lm_head 的
    ForCausalLM，直接调它返回的是 logits 而不是 hidden——分块 loss 的
    维度会对不上（441×152064 @ 5120×152064）。先 get_base_model 剥回
    ForCausalLM，再取其 .model 进 decoder。LoRA 层挂在 decoder 内部的
    Linear 上，直接调 decoder 前向 LoRA 照常生效。
    """
    try:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            model = model.get_base_model()
    except ImportError:
        pass
    return model.model if hasattr(model, "model") else model


def _forward_hidden(model, batch, device):
    input_ids = batch.input_ids.to(device)
    position_ids = batch.position_ids.to(device)
    if isinstance(model, TinyCausalLM):
        if batch.packed:
            return model(input_ids, batch.cu_seqlens.to(device), position_ids)
        return model(input_ids, None, position_ids, padding_mask=batch.padding_mask.to(device))
    inner = _decoder_of(model)
    kwargs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "use_cache": False,
    }
    if not batch.packed:
        kwargs["attention_mask"] = batch.padding_mask.to(device)
        outputs = inner(**kwargs)
    elif _supports_varlen(model):
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


def _apply_lora(model, cfg: TrainConfig):
    """lora_r > 0 时给 HF 模型挂 LoRA。TinyCausalLM 不挂。"""
    if cfg.lora_r <= 0:
        return model
    if isinstance(model, TinyCausalLM):
        logger.info("LoRA：tiny 模型跳过")
        return model
    from peft import LoraConfig, get_peft_model

    targets = [name.strip() for name in cfg.lora_target_modules.split(",") if name.strip()]
    config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _build_scheduler(optimizer, cfg: TrainConfig):
    """lr 调度：直接用 transformers 官方实现（和 HF Trainer 同源，经过大规模验证）。

    cosine = 线性 warmup + 余弦退火到 0；none = 恒定 lr。
    """
    if cfg.lr_scheduler == "none" or cfg.max_steps <= 0:
        return None
    from transformers import get_cosine_schedule_with_warmup

    # warmup 支持绝对步数或总步数比例两种给法，取大者，上限留一步
    warmup = max(cfg.warmup_steps, round(cfg.warmup_ratio * cfg.max_steps))
    warmup = max(0, min(warmup, cfg.max_steps - 1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, cfg.max_steps)
    logger.info("lr 调度：cosine（官方实现），warmup %s 步，%s 步退火到 0", warmup, cfg.max_steps)
    return scheduler


@torch.no_grad()
def _validation_loss(model, val_sequences, cfg: TrainConfig, accelerator, restriction) -> float:
    """在验证集上算平均 loss：纯前向、不反传、不动优化器。

    每条样本单独走一步（复用训练的 collate_plan + 分块 loss，同口径可比）。
    验证集限制在 512 条内，防止验证本身耗时过长。
    """
    from .data.adaptive import PlannedStep as _Step

    was_training = model.training
    model.eval()
    device = accelerator.device
    total, total_tokens = 0.0, 0
    try:
        for sequence in val_sequences:
            batch = collate_plan([sequence], _Step((0,), False, len(sequence), len(sequence)))
            labels = batch.labels.to(device)
            hidden = _forward_hidden(model, batch, device)
            weight = _lm_weight(model)
            if restriction is not None:
                weight = restriction.project(weight)
                labels = restriction.remap(labels)
            loss = chunked_causal_lm_loss(hidden, weight, labels, cfg.chunk_tokens)
            total += float(loss) * max(batch.content_tokens - 1, 1)
            total_tokens += max(batch.content_tokens - 1, 1)
    finally:
        if was_training:
            model.train()
    return total / max(total_tokens, 1)


def _build_model(cfg: TrainConfig, accelerator) -> torch.nn.Module:
    if cfg.model_name_or_path == "tiny":
        return TinyCausalLM(cfg.vocab_size, cfg.tiny_hidden, cfg.tiny_layers, cfg.tiny_heads)
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    dtype = torch.float32 if accelerator.kind == "cpu" else torch.bfloat16
    attn = pick_attn_implementation() if accelerator.kind != "cpu" else "sdpa"
    logger.info("加载 HuggingFace 模型 %s，注意力 %s", cfg.model_name_or_path, attn)
    kwargs = dict(torch_dtype=dtype, attn_implementation=attn, trust_remote_code=True)
    try:
        return AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **kwargs)
    except ValueError:
        # VL / 多模态架构不在 CausalLM 映射里，回退到图文模型类；纯文本序列同样能训
        logger.info("CausalLM 不识别该架构，按多模态模型加载")
        return AutoModelForImageTextToText.from_pretrained(cfg.model_name_or_path, **kwargs)


def _cpu_clock() -> float:
    import time

    return time.perf_counter()
