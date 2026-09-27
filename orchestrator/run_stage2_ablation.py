"""Stage2 消融。每组从同一份初始权重出发，不保存模型。

路径一律用环境变量，避免把机器路径写进仓库：

    export ORCH_MODEL=/path/to/Qwen3-VL-2B
    export ORCH_DATA=/path/to/stage2.json
    export ORCH_IMAGES=/path/to/images
    export ORCH_OUT=./runs
    export ABLATION_STEPS=10
    export ACCELERATOR=gpu
    python -m orchestrator.run_stage2_ablation

词表裁剪：python -m orchestrator.run_stage2_ablation --crop-vocab
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from orchestrator.backend import resolve_accelerator
from orchestrator.compute.attention import pick_attn_implementation
from orchestrator.stage2_experiment import Ablation, encode_stage2, run_ablation


def _required_path(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"请设置环境变量 {name}，指向模型或数据路径。见 README。")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage2 消融")
    parser.add_argument(
        "--crop-vocab",
        action="store_true",
        help="交叉熵只保留中文、英文和中英文符号，丢掉法语等其他语言的 token",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    model_path = _required_path("ORCH_MODEL")
    data_path = _required_path("ORCH_DATA")
    image_dir = _required_path("ORCH_IMAGES")
    out = Path(os.environ.get("ORCH_OUT", "./runs"))
    steps = int(os.environ.get("ABLATION_STEPS", "20"))
    device_name = os.environ.get("ACCELERATOR", "gpu")
    accelerator = resolve_accelerator(device_name)
    logging.info("加速器 %s，设备 %s，词表裁剪 %s", accelerator.kind, accelerator.device, args.crop_vocab)
    ablations = [
        Ablation("baseline", packing=False, precision="bf16", fused_adam=False, chunk_tokens=0, max_steps=steps, device=device_name),
        Ablation("packing", packing=True, precision="bf16", fused_adam=False, chunk_tokens=0, max_steps=steps, device=device_name),
        Ablation("fp8", packing=False, precision="fp8", fused_adam=False, chunk_tokens=0, max_steps=steps, device=device_name),
        Ablation("fused_adam", packing=False, precision="bf16", fused_adam=True, chunk_tokens=0, max_steps=steps, device=device_name),
        Ablation("chunked_loss", packing=False, precision="bf16", fused_adam=False, chunk_tokens=2048, max_steps=steps, device=device_name),
        Ablation("smart", packing=True, precision="bf16", fused_adam=False, chunk_tokens=2048, max_steps=steps, device=device_name),
        Ablation("all_on", packing=True, precision="fp8", fused_adam=True, chunk_tokens=2048, max_steps=steps, device=device_name),
    ]
    for ablation in ablations:
        ablation.restrict_vocab = args.crop_vocab
    cache = out / "encoded_stage2.pt"
    attn = pick_attn_implementation() if accelerator.kind != "cpu" else "sdpa"
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
        trust_remote_code=True,
    )
    if cache.exists():
        samples = torch.load(cache, weights_only=False)
        logging.info("复用已编码样本 %s", len(samples))
    else:
        samples = encode_stage2(model_path, data_path, image_dir, max_samples=160, model=model)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(samples, cache)
    initial = copy.deepcopy(model.state_dict())
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True) if args.crop_vocab else None
    summaries = []
    for ablation in ablations:
        logging.info("开始 %s", ablation.name)
        model.load_state_dict(initial)
        accelerator.empty_cache()
        summary = run_ablation(ablation, samples, model, out / ablation.name, tokenizer)
        summaries.append(summary)
        logging.info("结束 %s %s", ablation.name, json.dumps(summary, ensure_ascii=False))
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
