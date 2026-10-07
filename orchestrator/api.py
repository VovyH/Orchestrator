"""一键微调：给模型和数据，跑完直接存盘。

    from orchestrator import fit

    result = fit(
        "/mnt/WYH/models/Qwen2.5-0.5B",
        dataset_path="train.jsonl",
        output_dir="runs/qwen05b",
        epochs=1,
    )

数据支持三种行格式（jsonl），或 txt 一行一条：

    {"text": "..."}                     纯文本续训
    {"input_ids": [12, 34, ...]}        预先分好词
    {"messages": [{"role": ..., "content": ...}, ...]}   多轮，走 chat template

命令行等效，且不要求在项目根目录：

    python orchestrator/api.py --model ... --data ... --out ...
"""

from __future__ import annotations

if __package__ in (None, ""):  # 允许 python orchestrator/api.py 直接跑，不要求 cwd 在项目根
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "orchestrator"

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import TrainConfig
from .data.adaptive import normalize_packing, plan_steps
from .metrics import StepMeter
from .trainer import TrainResult, train_model

logger = logging.getLogger("orchestrator")


@dataclass
class FitResult:
    """fit() 的返回值。模型已经训完，output_dir 有就可以直接 from_pretrained。"""

    model: object
    output_dir: str | None
    meters: list[StepMeter]
    steps_per_epoch: int
    samples: int
    truncated: int
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
            "output_dir": self.output_dir,
            "steps": len(self.meters),
            "steps_per_epoch": self.steps_per_epoch,
            "samples": self.samples,
            "truncated_samples": self.truncated,
            "dropped_samples": self.dropped,
            "final_loss": self.final_loss,
            "tokens_per_second": round(self.tokens_per_second, 1),
            "peak_memory_mb": round(max(peaks)) if peaks else None,
        }

    def format(self) -> str:
        item = self.summary()
        return (
            f"步数={item['steps']}（每轮 {item['steps_per_epoch']}）样本={item['samples']} "
            f"截断={item['truncated_samples']} 丢弃={item['dropped_samples']} "
            f"末步 loss={item['final_loss']:.4f} tok/s={item['tokens_per_second']} "
            f"峰值={item['peak_memory_mb']} MiB 输出={item['output_dir']}"
        )


def fit(
    model_name_or_path: str,
    dataset_path: str | None = None,
    texts: list[str] | None = None,
    output_dir: str | None = None,
    *,
    max_seq_len: int = 1024,
    max_samples: int = 0,
    epochs: float = 1.0,
    max_steps: int | None = None,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    lr_scheduler: str = "cosine",
    warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    lora_r: int = 0,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    grad_clip: float = 1.0,
    batch_size: int = 2,
    packing: str = "on",
    pack_length_limit: int = 4096,
    bucket_ratio: float = 1.25,
    chunk_tokens: int = 2048,
    restrict_vocab: bool = False,
    precision: str = "bf16",
    fused_adam: bool = True,
    compile_mlp: bool = False,
    compile_attention: bool = False,
    checkpoint_every_n: int = 0,
    device: str = "auto",
    seed: int = 42,
    log_every: int = 10,
    val_dataset_path: str | None = None,
    early_stop_patience: int = 3,
    early_stop_every: int = 100,
    save_every: int = 0,
    resume: bool = True,
    truncate_long: bool = True,
    chat_template: bool = True,
    progress_bar: bool = True,
    tokenizer=None,
) -> FitResult:
    """训练一个因果语言模型，默认把结果存到 output_dir。

    packing="on" 是默认值：装了 flash-attn 就走 varlen 拼接，没装会自动退回并排。
    epochs 按调度器算出来的每轮步数换算成 max_steps；给了 max_steps 就以它为准。
    truncate_long：true 默认截断长句，但会保留对话结构。false 默认不截断，但会丢失数据。
    max_samples：最多取多少条样本参与训练。
    pack_length_limit：拼接时，每个窗口的长度不能超过这个值。
    bucket_ratio：分桶的松紧度，长短样本能不能进同一步。按长度排序切桶，桶内最长不能超过最短的这个倍数。
    chunk_tokens：词表分块大小，管 lm_head 算交叉熵时一次算多少 token 的 logits，是这个项目省显存的关键。
    """
    if dataset_path is None and texts is None:
        raise ValueError("dataset_path 和 texts 至少给一个")

    # 加载 Tokenizer
    if tokenizer is None and model_name_or_path != "tiny":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True) 

    # 加载训练数据
    rows = texts if texts is not None else _read_rows(dataset_path) 
    sequences, truncated, dropped = encode_rows(
        rows,
        tokenizer,
        max_seq_len=max_seq_len,
        max_samples=max_samples,
        truncate_long=truncate_long,
        chat_template=chat_template,
    )
    if not sequences:
        raise RuntimeError("没有可训练的样本：数据为空，或全部超长被丢弃")
    logger.info("数据：%s 条样本，截断 %s，丢弃 %s，最长 %s", len(sequences), truncated, dropped, max_seq_len)

    mode = normalize_packing(packing)
    layout = plan_steps(
        [len(sequence) for sequence in sequences],
        max_seq_len,
        mode,
        batch_size,
        0.10,
        pack_length_limit,
        bucket_ratio,
    )
    steps_per_epoch = max(1, len(layout.steps))
    total_steps = int(max_steps) if max_steps else max(1, round(epochs * steps_per_epoch))

    cfg = TrainConfig(
        model_name_or_path=model_name_or_path,
        dataset_path=dataset_path or "",
        max_seq_len=max_seq_len,
        max_samples=max_samples,
        max_steps=total_steps,
        lr=lr,
        weight_decay=weight_decay,
        lr_scheduler=lr_scheduler,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        grad_clip=grad_clip,
        nopack_batch_size=batch_size,
        packing=mode,
        pack_length_limit=pack_length_limit,
        bucket_ratio=bucket_ratio,
        chunk_tokens=chunk_tokens,
        restrict_vocab=restrict_vocab,
        precision=precision,
        fused_adam=fused_adam,
        compile_mlp=compile_mlp,
        compile_attention=compile_attention,
        checkpoint_every_n=checkpoint_every_n,
        device=device,
        seed=seed,
        log_every=log_every,
        progress_bar=progress_bar,
        val_dataset_path=val_dataset_path or "",
        early_stop_patience=early_stop_patience if val_dataset_path else 0,
        early_stop_every=early_stop_every,
        save_every=save_every,
        resume_from=str(output_dir) if (resume and output_dir) else "",
    )
    logger.info("计划：每轮 %s 步，共训 %s 步（epochs=%s），packing=%s", steps_per_epoch, total_steps, epochs, mode)

    file_handler = None
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(output_dir) / "train.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(file_handler)
    try:
        outcome: TrainResult = train_model(cfg, sequences=sequences, tokenizer=tokenizer)
    finally:
        if file_handler is not None:
            logger.removeHandler(file_handler)
            file_handler.close()
    if output_dir:
        _save(outcome.model, tokenizer, output_dir)
        _write_metrics(output_dir, cfg, outcome)

    return FitResult(
        model=outcome.model,
        output_dir=str(output_dir) if output_dir else None,
        meters=outcome.meters,
        steps_per_epoch=steps_per_epoch,
        samples=len(sequences),
        truncated=truncated,
        dropped=dropped,
    )


def encode_rows(
    rows,
    tokenizer,
    *,
    max_seq_len: int,
    max_samples: int = 0,
    truncate_long: bool = True,
    chat_template: bool = True,
) -> tuple[list[list[int]], int, int]:
    """把原始行变成 token id 序列。返回（序列，截断条数，丢弃条数）。"""
    sequences: list[list[int]] = []
    truncated = 0
    dropped = 0
    for row in rows:
        ids = row_to_ids(row, tokenizer, chat_template=chat_template)
        if not ids:
            continue
        if len(ids) > max_seq_len:
            if truncate_long:
                ids = ids[:max_seq_len]
                truncated += 1
            else:
                dropped += 1
                continue
        sequences.append(ids)
        if max_samples and len(sequences) >= max_samples:
            break
    return sequences, truncated, dropped


def row_to_ids(row, tokenizer, *, chat_template: bool = True) -> list[int]:
    """一行数据 → token id。dict 认 input_ids / text / messages，str 当纯文本。"""
    if isinstance(row, dict):
        if "input_ids" in row:
            return [int(token) for token in row["input_ids"]]
        if "text" in row:
            ids = _encode(tokenizer, row["text"])
        elif "messages" in row:
            ids = _encode_messages(tokenizer, row["messages"], chat_template)
        else:
            raise ValueError("每行需要 input_ids、text 或 messages 之中的一个")
    else:
        ids = _encode(tokenizer, str(row))

    end = getattr(tokenizer, "eos_token_id", None)
    if end is not None and (not ids or ids[-1] != end):
        ids = ids + [int(end)]
    return ids


def _encode(tokenizer, text: str) -> list[int]:
    if tokenizer is None:
        raise ValueError("纯文本需要 tokenizer：换成有 tokenizer 的模型，或直接给 input_ids")
    return list(tokenizer.encode(text, add_special_tokens=False))


def _encode_messages(tokenizer, messages, chat_template: bool) -> list[int]:
    if tokenizer is None:
        raise ValueError("messages 需要 tokenizer：换成有 tokenizer 的模型，或直接给 input_ids")
    if chat_template and getattr(tokenizer, "chat_template", None):
        # transformers 5.x 起 apply_chat_template 默认 return_dict=True，返回的是
        # BatchEncoding（dict），list() 它只会拿到 key 字符串——必须显式关掉。
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=False
        )
        if isinstance(out, dict):  # 兜底：个别版本仍返回 dict，取 input_ids
            out = out["input_ids"]
        return [int(token) for token in out]
    joined = "\n".join(f"{item.get('role', '')}: {item.get('content', '')}" for item in messages)
    return _encode(tokenizer, joined)


def _read_rows(path: str) -> list:
    """jsonl 每行一个对象；.txt / 其它后缀按纯文本一行一条。"""
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".jsonl") or path.endswith(".json"):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [line for line in text.splitlines() if line.strip()]


def _save(model, tokenizer, output_dir: str) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            merged = model.merge_and_unload()  # LoRA 并回主权重，eval/部署直接 from_pretrained
            merged.save_pretrained(directory, safe_serialization=True)
            logger.info("LoRA 已合并进主权重")
            model = merged
    except ImportError:
        pass
    model.save_pretrained(directory, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(directory)
    logger.info("模型已存到 %s", directory)


def _write_metrics(output_dir: str, cfg: TrainConfig, outcome: TrainResult) -> None:
    payload = outcome.summary()
    payload["config"] = {
        "model_name_or_path": cfg.model_name_or_path,
        "dataset_path": cfg.dataset_path,
        "max_seq_len": cfg.max_seq_len,
        "lr": cfg.lr,
        "lr_scheduler": cfg.lr_scheduler,
        "warmup_steps": cfg.warmup_steps,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha if cfg.lora_r > 0 else None,
        "packing": cfg.packing,
        "chunk_tokens": cfg.chunk_tokens,
        "precision": cfg.precision,
    }
    payload["step_log"] = [
        {
            "step": meter.step,
            "loss": meter.loss,
            "tokens": meter.content_tokens,
            "tok_s": round(meter.tokens_per_second, 1),
            "waste": round(meter.padding_waste, 4),
            "peak_mb": None if meter.peak_memory_mb is None else round(meter.peak_memory_mb),
            "grad_norm": None if meter.grad_norm is None else round(meter.grad_norm, 6),
            "gpu_util": None if meter.gpu_util is None else round(meter.gpu_util, 1),
            "gpu_memory_mb": None if meter.gpu_memory_mb is None else round(meter.gpu_memory_mb),
        }
        for meter in outcome.meters
    ]
    (Path(output_dir) / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="一键微调一个因果语言模型")
    parser.add_argument("--model", required=True, help="模型目录或 hub id")
    parser.add_argument("--data", help="jsonl（text / input_ids / messages）或 txt")
    parser.add_argument("--out", help="输出目录，给了就存盘")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="总步数的比例，和 --warmup-steps 取大者")
    parser.add_argument("--lora-r", type=int, default=0, help=">0 启用 LoRA，0 = 全参")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--packing", choices=["auto", "on", "off"], default="on")
    parser.add_argument("--chunk-tokens", type=int, default=2048)
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "auto"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-every-n", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-data", default=None, help="验证集 jsonl，用于早停")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0, help=">0 每隔这么多步存 checkpoint")
    parser.add_argument("--no-resume", action="store_true", help="不给输出目录恢复断点，从头训")
    parser.add_argument("--crop-vocab", action="store_true", help="交叉熵只留中文、英文和符号")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    result = fit(
        args.model,
        dataset_path=args.data,
        output_dir=args.out,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        packing=args.packing,
        chunk_tokens=args.chunk_tokens,
        restrict_vocab=args.crop_vocab,
        precision=args.precision,
        device=args.device,
        checkpoint_every_n=args.checkpoint_every_n,
        log_every=args.log_every,
        val_dataset_path=args.val_data,
        early_stop_patience=args.early_stop_patience,
        early_stop_every=args.early_stop_every,
        save_every=args.save_every,
        resume=not args.no_resume,
    )
    logger.info(result.format())


if __name__ == "__main__":
    main()
