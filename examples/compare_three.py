"""三条曲线对比：OR vs LF vs LF-pack 同坐标系叠画，loss / grad_norm 各一张。

    python examples/compare_three.py \
        --log-or <metrics.json> --log-lf <trainer_state.json> --log-lfpack <trainer_state.json> \
        --out /mnt/WYH/runs/plots/compare --prefix ev- --window 50
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse

from compare_runs import _load, _ma_records
from plot_lf_metrics import moving_average


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-or", required=True, help="OR 的 metrics.json")
    parser.add_argument("--log-lf", required=True, help="LF 的 trainer_state.json / trainer_log.jsonl")
    parser.add_argument("--log-lfpack", required=True, help="LF-pack 的同 LF")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--prefix", required=True, help="文件名前缀，如 ev-")
    parser.add_argument("--window", type=int, default=50, help="MA 平滑窗口（step），按各 run 日志间隔自动换算条数")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for label, log, color in (
        ("OR", args.log_or, "#d62728"),
        ("LF", args.log_lf, "#1f77b4"),
        ("LF-pack", args.log_lfpack, "#2ca02c"),
    ):
        run = _load(Path(log))
        run["label"], run["color"] = label, color
        run["ma"] = _ma_records(run["steps"], args.window)
        runs.append(run)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-window{args.window}"

    def decorate(ax) -> None:
        ax.legend(loc="upper right", frameon=True, edgecolor="black",
                  facecolor="white", framealpha=1.0, fontsize=9)
        ax.grid(alpha=0.3)

    # ---- 图 1：loss 三方对比 ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for run in runs:
        ax.plot(run["steps"], run["loss"], color=run["color"], alpha=0.25, lw=1.5,
                label=f"{run['label']} (raw)")
        ax.plot(run["steps"], moving_average(run["loss"], run["ma"]), color=run["color"], lw=3.5,
                label=f"{run['label']} (MA-{args.window} steps)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    decorate(ax)
    fig.tight_layout()
    loss_png = out_dir / f"{args.prefix}three-way-compare-loss{suffix}.png"
    fig.savefig(loss_png, dpi=150)
    plt.close(fig)
    print(f"已保存 {loss_png}")

    # ---- 图 2：grad_norm 三方对比 ----
    if any(run["grad"] for run in runs):
        fig, ax = plt.subplots(figsize=(10, 5))
        for run in runs:
            if not run["grad"]:
                continue
            ax.plot(run["grad_steps"], run["grad"], color=run["color"], alpha=0.25, lw=1.5,
                    label=f"{run['label']} (raw)")
            ax.plot(run["grad_steps"], moving_average(run["grad"], run["ma"]), color=run["color"], lw=3.5,
                    label=f"{run['label']} (MA-{args.window} steps)")
        ax.set_xlabel("step")
        ax.set_ylabel("grad_norm")
        ax.set_yscale("log")
        decorate(ax)
        fig.tight_layout()
        grad_png = out_dir / f"{args.prefix}three-way-compare-grad{suffix}.png"
        fig.savefig(grad_png, dpi=150)
        plt.close(fig)
        print(f"已保存 {grad_png}")

    for run in runs:
        ma_loss = moving_average(run["loss"], run["ma"])[-1]
        line = f"{run['label']}: {len(run['loss'])} 条，loss 末值 {run['loss'][-1]:.4f}，MA 末值 {ma_loss:.4f}"
        if run["grad"]:
            line += f"；grad MA 末值 {moving_average(run['grad'], run['ma'])[-1]:.3f}"
        print(line)


if __name__ == "__main__":
    main()
