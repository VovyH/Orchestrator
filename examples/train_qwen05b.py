"""用 Qwen2.5-0.5B 跑一次序列微调（因果语言建模）。

    python examples/train_qwen05b.py          # 在项目根目录
    python /path/to/Orchestrator/examples/train_qwen05b.py   # 任意目录

数据集是 messages 格式的 jsonl，走 tokenizer 自带的 chat template。
换成自己的模型和数据，改下面三个常量就行。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根目录进 sys.path

from orchestrator import fit  # noqa: E402

MODEL = "/mnt/WYH/models/Qwen2.5-0.5B"
DATA = "/mnt/WYH/data/ev-charging-protocols/train.jsonl"
OUTPUT = "/mnt/WYH/runs/qwen2.5-0.5b-orchestrator"


def main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = fit(
        MODEL,
        dataset_path=DATA,
        output_dir=OUTPUT,
        max_seq_len=4096,
        max_samples=0,
        epochs=1,
        lr=1e-4,
        batch_size=4,
        packing="on",
        chunk_tokens=1024,
        log_every=1,
    )
    print(result.format())


if __name__ == "__main__":
    main()
