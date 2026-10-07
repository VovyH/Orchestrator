"""按长度排序后，用双指针把变长样本装进固定窗口。

右指针取当前最长、且不超过窗口的样本，左指针用最短样本填满剩余空隙。
窗口装不下的样本直接丢弃，不截断，避免标签和原文错位。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PackedWindow:
    sample_ids: tuple[int, ...]
    tokens: int


def pack_by_length(lengths: list[int], max_seq_len: int) -> tuple[list[PackedWindow], list[int]]:
    if max_seq_len < 1:
        raise ValueError("max_seq_len 必须为正数")

    order = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    left = 0
    right = len(order) - 1
    windows: list[PackedWindow] = []
    dropped: list[int] = []

    while left <= right:
        longest = order[right]
        if lengths[longest] > max_seq_len:
            dropped.append(longest)
            right -= 1
            continue

        chosen = [longest]
        used = lengths[longest]
        right -= 1
        while left <= right and used + lengths[order[left]] <= max_seq_len:
            chosen.append(order[left])
            used += lengths[order[left]]
            left += 1
        windows.append(PackedWindow(tuple(chosen), used))

    return windows, dropped


def cutoff_slack(windows: list[PackedWindow], max_seq_len: int) -> float:
    """相对「每个窗口都垫到 max_seq_len」的空隙比例。张量本身不再另加 pad。"""
    if not windows:
        return 0.0
    content = sum(window.tokens for window in windows)
    slots = len(windows) * max_seq_len
    return 1.0 - content / slots
