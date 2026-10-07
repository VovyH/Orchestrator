"""读 LLaMA-Factory 的训练日志，画 loss 和 grad_norm 各一张 png。

数据源二选一（推荐 trainer_state.json，有 grad_norm）：
    python examples/plot_lf_metrics.py --log /mnt/WYH/runs/qwen25-14b-lf/trainer_state.json
    python examples/plot_lf_metrics.py --log /mnt/WYH/runs/qwen25-14b-lf/trainer_log.jsonl  # 自动转 state
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json


def moving_average(values, window: int) -> list[float]:
    half = max(1, window // 2)
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def load_records(path: Path) -> list[dict]:
    """读出 [{step, loss, grad_norm}]，步数倒退视为新一轮训练，只保留最后一段。"""
    if path.name == "trainer_state.json":
        # HF Trainer 的标准状态文件：log_history 有 loss + grad_norm + lr 全套
        state = json.loads(path.read_text(encoding="utf-8"))
        rows = [
            {"current_steps": h["step"], "loss": h["loss"], "grad_norm": h.get("grad_norm")}
            for h in state["log_history"]
            if "loss" in h
        ]
    else:
        # trainer_log.jsonl 只有 loss/lr，没有 grad_norm（LF 只把 grad 打到终端）
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "loss" in row and "current_steps" in row:
                rows.append({"current_steps": row["current_steps"],
                             "loss": row["loss"], "grad_norm": row.get("grad_norm")})
        if all(r.get("grad_norm") is None for r in rows):
            sibling = path.parent / "trainer_state.json"
            if sibling.is_file():
                print("trainer_log.jsonl 里没有 grad_norm，改读同目录的 trainer_state.json")
                return load_records(sibling)

    # 同一个 output_dir 可能被多次训练追加过（试跑/中断重跑），
    # 步数一旦倒退就是新一轮的开始——只保留最后一段
    kept = []
    prev_step = -1
    for row in rows:
        if row["current_steps"] <= prev_step:
            kept = []  # 新一轮，清掉之前段
        kept.append(row)
        prev_step = row["current_steps"]
    if len(kept) < len(rows):
        print(f"注意：日志含 {len(rows) - len(kept)} 条旧训练记录（试跑/中断），已只保留最后一轮")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="LF 的 trainer_log.jsonl 或 trainer_state.json")
    parser.add_argument("--out", default=None, help="图片输出目录，默认 log 同目录")
    parser.add_argument("--smooth", type=int, default=50, help="平滑窗口（logging_steps=1，窗口即步数）")
    parser.add_argument("--prefix", default=None,
                        help="输出文件名前缀：有前缀时文件名为 <prefix>loss-window<N>.png / <prefix>grad-window<N>.png")
    parser.add_argument("--label", default=None, help="图例里的 run 名，如 qwen25-14B-lf")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_records(Path(args.log))
    steps = [r["current_steps"] for r in records]
    loss = [float(r["loss"]) for r in records]
    grad = [float(r["grad_norm"]) for r in records if r.get("grad_norm") is not None]
    grad_steps = [r["current_steps"] for r in records if r.get("grad_norm") is not None]

    out_dir = Path(args.out) if args.out else Path(args.log).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    lp = f"{args.label} " if args.label else ""
    # ---- 图 1：loss ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, loss, color="#1f77b4", alpha=0.3, lw=1.8, label=f"{lp}loss (raw)")
    ax.plot(steps, moving_average(loss, args.smooth), color="#d62728", lw=3.5,
            label=f"{lp}loss (MA-{args.smooth})")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right", fontsize=8, frameon=True, edgecolor="black", facecolor="white", framealpha=1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    suffix = f"-window{args.smooth}" if args.prefix else ""
    loss_png = out_dir / (f"{args.prefix}loss{suffix}.png" if args.prefix else "loss.png")
    fig.savefig(loss_png, dpi=150)
    plt.close(fig)

    # ---- 图 2：grad_norm ----
    grad_png = None
    if grad:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grad_steps, grad, color="#2ca02c", alpha=0.3, lw=1.8, label=f"{lp}grad_norm (raw)")
        ax.plot(grad_steps, moving_average(grad, args.smooth), color="#9467bd", lw=3.5,
                label=f"{lp}grad_norm (MA-{args.smooth})")
        ax.set_xlabel("step")
        ax.set_ylabel("grad_norm")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8, frameon=True, edgecolor="black", facecolor="white", framealpha=1.0)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        grad_png = out_dir / (f"{args.prefix}grad{suffix}.png" if args.prefix else "grad.png")
        fig.savefig(grad_png, dpi=150)
        plt.close(fig)

    print(f"已保存 {loss_png}")
    if grad_png:
        print(f"已保存 {grad_png}")
    print(f"记录 {len(records)} 条；loss 末值 {loss[-1]:.4f}，MA 末值 {moving_average(loss, args.smooth)[-1]:.4f}")
    if grad:
        print(f"grad 末值 {grad[-1]:.3f}，MA 末值 {moving_average(grad, args.smooth)[-1]:.3f}")


if __name__ == "__main__":
    main()
