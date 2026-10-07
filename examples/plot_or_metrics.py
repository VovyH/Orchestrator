"""读 Orchestrator 的 metrics.json，画 loss 和 grad_norm 各一张 png（风格与 plot_lf_metrics.py 对齐）。

    python examples/plot_or_metrics.py --metrics /mnt/WYH/runs/qwen25-14b-orch/metrics.json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="Orchestrator 的 metrics.json")
    parser.add_argument("--out", default=None, help="图片输出目录，默认 metrics 同目录")
    parser.add_argument("--smooth", type=int, default=10, help="移动平均窗口")
    parser.add_argument("--prefix", default=None,
                        help="输出文件名前缀：有前缀时文件名为 <prefix>loss-window<N>.png / <prefix>grad-window<N>.png")
    parser.add_argument("--label", default=None, help="图例里的 run 名，如 qwen25-14B-orch")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    log = data["step_log"]
    steps = [item["step"] for item in log]
    loss = [float(item["loss"]) for item in log]
    grad = [float(item["grad_norm"]) for item in log if item.get("grad_norm") is not None]
    grad_steps = [item["step"] for item in log if item.get("grad_norm") is not None]

    out_dir = Path(args.out) if args.out else Path(args.metrics).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    config = data.get("config", {})

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
    print(f"已保存 {grad_png}")
    print(f"记录 {len(log)} 条；loss 末值 {loss[-1]:.4f}，MA 末值 {moving_average(loss, args.smooth)[-1]:.4f}")
    if grad:
        print(f"grad 末值 {grad[-1]:.3f}，MA 末值 {moving_average(grad, args.smooth)[-1]:.3f}")


if __name__ == "__main__":
    main()
