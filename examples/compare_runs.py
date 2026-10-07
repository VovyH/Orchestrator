"""对比两次训练的 loss / grad_norm：各出一张图，同坐标系叠画。

支持三种日志（自动识别）：
    - LLaMA-Factory: trainer_state.json / trainer_log.jsonl（每10步一条）
    - Orchestrator:  metrics.json 的 step_log（每步一条）

用法：
    python examples/compare_runs.py \
        --log-a /mnt/WYH/runs/qwen25-14b-lf/trainer_state.json \
        --log-b /mnt/WYH/runs/qwen25-14b-orch/metrics.json

--window 的单位是 step（默认 50），会按每个 run 自己的日志间隔换算成条数，
所以两边 MA 线的平滑程度是可比的。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

from plot_lf_metrics import load_records, moving_average


def _load(path: Path) -> dict:
    """读一个 run，拆成 loss 序列和 grad 序列（grad 可能缺失）。"""
    if path.name == "metrics.json":
        # Orchestrator 的逐步日志：step_log 里每步一条
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [
            {"current_steps": s["step"], "loss": s["loss"], "grad_norm": s.get("grad_norm")}
            for s in data.get("step_log", [])
            if "loss" in s
        ]
    else:
        rows = load_records(path)
    grad_rows = [r for r in rows if r.get("grad_norm") is not None]
    return {
        "label": path.parent.name,
        "steps": [r["current_steps"] for r in rows],
        "loss": [float(r["loss"]) for r in rows],
        "grad_steps": [r["current_steps"] for r in grad_rows],
        "grad": [float(r["grad_norm"]) for r in grad_rows],
    }


def _ma_records(steps: list[int], window_steps: int) -> int:
    """把以 step 为单位的平滑窗口换算成该序列的条数。"""
    if len(steps) < 2:
        return 1
    interval = (steps[-1] - steps[0]) / (len(steps) - 1)
    return max(1, round(window_steps / max(interval, 1e-9)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-a", required=True, help="运行 A 的日志（trainer_state.json / trainer_log.jsonl / metrics.json）")
    parser.add_argument("--log-b", required=True, help="运行 B 的同上")
    parser.add_argument("--label-a", default=None, help="图例文字，默认取日志父目录名")
    parser.add_argument("--label-b", default=None)
    parser.add_argument("--out", default=None, help="输出目录，默认 log-a 同目录")
    parser.add_argument("--window", type=int, default=50, help="MA 平滑窗口（step），按各 run 日志间隔自动换算条数")
    parser.add_argument("--prefix", default=None,
                        help="输出文件名前缀：有前缀时为 <prefix>compare-loss-window<N>.png / <prefix>compare-grad-window<N>.png")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for log, label, color in (
        (args.log_a, args.label_a, "#1f77b4"),
        (args.log_b, args.label_b, "#d62728"),
    ):
        run = _load(Path(log))
        run["label"] = label or run["label"]
        run["color"] = color
        run["ma"] = _ma_records(run["steps"], args.window)
        runs.append(run)

    out_dir = Path(args.out) if args.out else Path(args.log_a).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    def decorate(ax) -> None:
        ax.legend(loc="upper right", frameon=True, edgecolor="black",
                  facecolor="white", framealpha=1.0)
        ax.grid(alpha=0.3)

    # ---- 图 1：loss 对比 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for run in runs:
        ax.plot(run["steps"], run["loss"], color=run["color"], alpha=0.3, lw=1.8,
                label=f"{run['label']} loss (raw)")
        ax.plot(run["steps"], moving_average(run["loss"], run["ma"]), color=run["color"],
                lw=3.5, label=f"{run['label']} loss (MA-{args.window} steps)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    decorate(ax)
    fig.tight_layout()
    suffix = f"-window{args.window}" if args.prefix else ""
    loss_png = out_dir / (f"{args.prefix}compare-loss{suffix}.png" if args.prefix else "compare_loss.png")
    fig.savefig(loss_png, dpi=150)
    plt.close(fig)

    # ---- 图 2：grad_norm 对比 ----
    grad_png = None
    if any(run["grad"] for run in runs):
        fig, ax = plt.subplots(figsize=(10, 5))
        for run in runs:
            if not run["grad"]:
                continue
            ax.plot(run["grad_steps"], run["grad"], color=run["color"], alpha=0.3, lw=1.8,
                    label=f"{run['label']} grad_norm (raw)")
            ax.plot(run["grad_steps"], moving_average(run["grad"], run["ma"]), color=run["color"],
                    lw=3.5, label=f"{run['label']} grad_norm (MA-{args.window} steps)")
        ax.set_xlabel("step")
        ax.set_ylabel("grad_norm")
        ax.set_yscale("log")
        decorate(ax)
        fig.tight_layout()
        grad_png = out_dir / (f"{args.prefix}compare-grad{suffix}.png" if args.prefix else "compare_grad.png")
        fig.savefig(grad_png, dpi=150)
        plt.close(fig)

    print(f"已保存 {loss_png}")
    if grad_png:
        print(f"已保存 {grad_png}")
    for run in runs:
        ma_loss = moving_average(run["loss"], run["ma"])[-1]
        line = (f"{run['label']}: {len(run['loss'])} 条（MA 窗口 {run['ma']} 条 = {args.window} 步），"
                f"loss 末值 {run['loss'][-1]:.4f}，MA 末值 {ma_loss:.4f}")
        if run["grad"]:
            ma_grad = moving_average(run["grad"], run["ma"])[-1]
            line += f"；grad 末值 {run['grad'][-1]:.3f}，MA 末值 {ma_grad:.3f}"
        print(line)


if __name__ == "__main__":
    main()
