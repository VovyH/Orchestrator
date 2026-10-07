"""读 metrics.json，画 loss 和 grad_norm 曲线（含移动平均）。

    python /mnt/WYH/projects/project-7/Orchestrator/examples/plot_metrics.py \
        --metrics /mnt/WYH/runs/qwen2.5-0.5b-orchestrator/metrics.json \
        --out /mnt/WYH/runs/qwen2.5-0.5b-orchestrator
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
    parser.add_argument("--metrics", required=True, help="metrics.json 路径")
    parser.add_argument("--out", default=None, help="图片输出目录，默认 metrics 同目录")
    parser.add_argument("--smooth", type=int, default=101, help="移动平均窗口（步）")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    log = data["step_log"]
    steps = [item["step"] for item in log]
    loss = [item["loss"] for item in log]
    grad = [item["grad_norm"] for item in log if item.get("grad_norm") is not None]
    grad_steps = [item["step"] for item in log if item.get("grad_norm") is not None]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(steps, loss, color="#1f77b4", alpha=0.25, lw=0.6, label="loss (raw)")
    axes[0].plot(steps, moving_average(loss, args.smooth), color="#d62728", lw=1.6,
                 label=f"loss (MA-{args.smooth})")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{data['config']['model_name_or_path']}  "
                      f"{data['samples']} samples, {data['steps']} steps, packing={data['config']['packing']}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(grad_steps, grad, color="#2ca02c", alpha=0.25, lw=0.6, label="grad_norm (raw)")
    axes[1].plot(grad_steps, moving_average(grad, args.smooth), color="#9467bd", lw=1.6,
                 label=f"grad_norm (MA-{args.smooth})")
    axes[1].axhline(1.0, color="gray", ls="--", lw=0.8, label="clip=1.0")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("grad_norm")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    out_dir = Path(args.out) if args.out else Path(args.metrics).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    figure = out_dir / "loss_grad.png"
    fig.tight_layout()
    fig.savefig(figure, dpi=150)
    print(f"已保存 {figure}")
    print(f"loss 末步 {loss[-1]:.4f}，MA-{args.smooth} 末值 {moving_average(loss, args.smooth)[-1]:.4f}")
    if grad:
        print(f"grad 末步 {grad[-1]:.3f}，MA-{args.smooth} 末值 {moving_average(grad, args.smooth)[-1]:.3f}")


if __name__ == "__main__":
    main()
