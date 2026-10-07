"""生成式评测：输入问题 → 模型生成回答 → 和参考答案比对打分。

    python examples/eval_qwen05b.py --max-samples 200
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import re

import torch

MODEL = "/mnt/WYH/runs/qwen2.5-0.5b-orchestrator"
DATA = "/mnt/WYH/data/ev-charging-protocols/val.jsonl"


def normalize(text: str) -> str:
    """比对前的归一化：去模板标记、空白、标点和大小写。"""
    text = re.sub(r"<\|.*?\|>", "", text)  # <|im_end|> 之类的残留
    text = re.sub(r"[\s`*_#>]+", " ", text)
    text = re.sub(r"[.,!?;:'\"()\[\]{}，。！？；：（）]+", "", text)
    return text.strip().lower()


def score(pred: str, ref: str) -> dict:
    n_pred, n_ref = normalize(pred), normalize(ref)
    contains = n_ref in n_pred
    # 词级 F1
    p_words, r_words = set(n_pred.split()), set(n_ref.split())
    tp = len(p_words & r_words)
    precision = tp / len(p_words) if p_words else 0.0
    recall = tp / len(r_words) if r_words else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"contains": contains, "f1": round(f1, 4), "precision": round(precision, 4),
            "recall": round(recall, 4)}


@torch.no_grad()
def generate_answers(model, tokenizer, rows, device, max_samples, max_new_tokens, batch_size, temperature=None, seed=None):
    from tqdm import tqdm

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    rows = rows[:max_samples] if max_samples else rows
    results = []
    for start in tqdm(range(0, len(rows), batch_size), unit="batch", dynamic_ncols=True):
        chunk = rows[start : start + batch_size]
        prompts, refs = [], []
        for row in chunk:
            messages = row["messages"]
            # 问题 = 除最后一条 assistant 外的全部
            question = [m for m in messages if not (m is messages[-1] and m["role"] == "assistant")]
            prompts.append(tokenizer.apply_chat_template(question, tokenize=False, add_generation_prompt=True))
            ref = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
            refs.append(ref)

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if temperature is not None:
            gen_kwargs.update(do_sample=True, temperature=temperature)  # 只用 temperature，不设 top_p
        else:
            gen_kwargs.update(do_sample=False)
        outputs = model.generate(**inputs, **gen_kwargs)
        texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for prompt, ref, text in zip(prompts, refs, texts):
            results.append({"prompt": prompt, "reference": ref, "prediction": text, "scores": score(text, ref)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = 全量")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=None,
                        help="不传 = 贪心解码；传了 = 采样（只用 temperature，不设 top_p）")
    parser.add_argument("--seed", type=int, default=42, help="采样模式的随机种子")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", default=None, help="结果输出目录，默认模型目录")
    parser.add_argument("--show-failures", type=int, default=5, help="打印几条最差的样例")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device).eval()

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"数据 {args.data}：{len(rows)} 条")

    results = generate_answers(
        model, tokenizer, rows, device, args.max_samples, args.max_new_tokens,
        args.batch_size, temperature=args.temperature, seed=args.seed,
    )

    n = len(results)
    contains_rate = sum(r["scores"]["contains"] for r in results) / n
    mean_f1 = sum(r["scores"]["f1"] for r in results) / n

    print(f"\n=== {n} 条评测结果 ===")
    print(f"答案包含率（参考答案完整出现在生成里）：{contains_rate:.1%}")
    print(f"平均词级 F1：{mean_f1:.4f}")

    if args.show_failures:
        worst = sorted(results, key=lambda r: r["scores"]["f1"])[: args.show_failures]
        print(f"\n--- 最差 {len(worst)} 条 ---")
        for r in worst:
            print(f"\n[F1={r['scores']['f1']}] 参考: {r['reference'][:120]}")
            print(f"          生成: {r['prediction'][:120]}")

    out_dir = Path(args.out) if args.out else Path(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "eval_gen.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n逐条结果已存 {out}")
    summary = {"samples": n, "contains_rate": contains_rate, "mean_f1": mean_f1}
    (out_dir / "eval_gen_summary.json").write_text(
        json.dumps({"model": args.model, "data": args.data, **summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
