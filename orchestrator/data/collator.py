"""把一个 step 拼成一条变长序列。

每个子序列的 position_ids 从 0 重新计数。
子序列交界处的标签写成 -100，这样上一段的最后一个 token 不会去预测下一段的开头。
"""

from dataclasses import dataclass

import torch

from .adaptive import PlannedStep
from .packing import PackedWindow
from .sampler import TokenStep


@dataclass
class PackedBatch:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    labels: torch.Tensor
    cu_seqlens: torch.Tensor
    content_tokens: int
    cutoff_slots: int
    packed: bool = True
    padding_mask: torch.Tensor | None = None


def collate_step(
    sequences: list[list[int]],
    windows: list[PackedWindow],
    step: TokenStep,
    max_seq_len: int,
) -> PackedBatch:
    tokens: list[int] = []
    positions: list[int] = []
    cuts = [0]
    for window_id in step.window_ids:
        window = windows[window_id]
        for sample_id in window.sample_ids:
            sequence = sequences[sample_id]
            tokens.extend(sequence)
            positions.extend(range(len(sequence)))
            cuts.append(cuts[-1] + len(sequence))

    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    position_ids = torch.tensor(positions, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    for boundary in cuts[1:-1]:
        labels[0, boundary] = -100

    return PackedBatch(
        input_ids=input_ids,
        position_ids=position_ids,
        labels=labels,
        cu_seqlens=torch.tensor(cuts, dtype=torch.int32),
        content_tokens=len(tokens),
        cutoff_slots=len(step.window_ids) * max_seq_len,
    )


def collate_plan(sequences: list[list[int]], step: PlannedStep) -> PackedBatch:
    """packing 步接成一条；batch 步并排，短的用 0 和 -100 垫到组内最长。"""
    chosen = [sequences[index] for index in step.sample_ids]
    if step.packed:
        return _collate_packed_sequences(chosen, step.slots)
    return _collate_padded_sequences(chosen)


def _collate_packed_sequences(chosen: list[list[int]], slots: int) -> PackedBatch:
    tokens: list[int] = []
    positions: list[int] = []
    cuts = [0]
    for sequence in chosen:
        tokens.extend(sequence)
        positions.extend(range(len(sequence)))
        cuts.append(cuts[-1] + len(sequence))
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    return PackedBatch(
        input_ids=input_ids,
        position_ids=torch.tensor(positions, dtype=torch.long).unsqueeze(0),
        labels=_boundary_labels(input_ids, cuts),
        cu_seqlens=torch.tensor(cuts, dtype=torch.int32),
        content_tokens=len(tokens),
        cutoff_slots=slots,
        packed=True,
    )


def _collate_padded_sequences(chosen: list[list[int]]) -> PackedBatch:
    width = max(len(sequence) for sequence in chosen)
    ids, labels, positions, mask = [], [], [], []
    content = 0
    for sequence in chosen:
        length = len(sequence)
        content += length
        pad = width - length
        ids.append(sequence + [0] * pad)
        labels.append(sequence + [-100] * pad)
        positions.append(list(range(length)) + [0] * pad)
        mask.append([1] * length + [0] * pad)
    input_ids = torch.tensor(ids, dtype=torch.long)
    return PackedBatch(
        input_ids=input_ids,
        position_ids=torch.tensor(positions, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        cu_seqlens=torch.tensor([0, content], dtype=torch.int32),
        content_tokens=content,
        cutoff_slots=width * len(chosen),
        packed=False,
        padding_mask=torch.tensor(mask, dtype=torch.long),
    )


def _boundary_labels(input_ids: torch.Tensor, cuts: list[int]) -> torch.Tensor:
    labels = input_ids.clone()
    for boundary in cuts[1:-1]:
        labels[0, boundary] = -100
    return labels
