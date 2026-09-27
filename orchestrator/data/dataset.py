"""读取预分词 jsonl，或在没有数据文件时生成固定种子的变长样本。"""

import json
from pathlib import Path

import torch

from ..config import TrainConfig


def load_sequences(cfg: TrainConfig, tokenizer=None) -> list[list[int]]:
    if not cfg.dataset_path:
        return synthetic_sequences(cfg)

    sequences: list[list[int]] = []
    for line in Path(cfg.dataset_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "input_ids" in row:
            sequences.append([int(token) for token in row["input_ids"]])
        elif "text" in row and tokenizer is not None:
            sequences.append(list(tokenizer.encode(row["text"])))
        else:
            raise ValueError("每一行需要 input_ids，或在提供 tokenizer 时给出 text")
        if cfg.max_samples and len(sequences) >= cfg.max_samples:
            break
    return sequences


def synthetic_sequences(cfg: TrainConfig) -> list[list[int]]:
    generator = torch.Generator().manual_seed(cfg.seed)
    low = max(1, cfg.synthetic_min_len)
    high = max(low, cfg.synthetic_max_len)
    sequences = []
    for _ in range(cfg.synthetic_samples):
        length = int(torch.randint(low, high + 1, (1,), generator=generator))
        tokens = torch.randint(1, cfg.vocab_size, (length,), generator=generator)
        sequences.append(tokens.tolist())
    return sequences
