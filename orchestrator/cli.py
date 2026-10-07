"""命令行入口：python -m orchestrator.cli --config orchestrator/configs/smoke.yaml

也可以不进项目根目录直接跑：python /path/to/Orchestrator/orchestrator/cli.py --config ...
"""

if __package__ in (None, ""):  # 直接以脚本运行时，把项目根目录放进 sys.path
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "orchestrator"

import argparse
import logging

from .config import load_config
from .trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(description="轻量训练调度")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--crop-vocab",
        action="store_true",
        help="交叉熵只保留中文、英文和中英文符号，丢掉法语等其他语言的 token",
    )
    parser.add_argument("--packing", choices=["auto", "on", "off"], help="auto 会试跑后决定走 packing 还是 batch")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(args.config)
    if args.crop_vocab:
        config.restrict_vocab = True
    if args.packing:
        config.packing = args.packing
    train(config)


if __name__ == "__main__":
    main()
