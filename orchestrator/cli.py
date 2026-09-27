"""命令行入口：python -m orchestrator.cli --config orchestrator/configs/smoke.yaml"""

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
