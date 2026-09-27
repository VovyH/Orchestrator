"""把已经装好的窗口按 token 预算收成一个 step。

不再使用固定样本数作为 batch。一个 step 里可以有多个窗口，只要 token 之和不超过预算。
单个窗口若已经超过预算，仍独占一步：窗口长度已被 max_seq_len 限制住。
"""

from dataclasses import dataclass

from .packing import PackedWindow


@dataclass(frozen=True)
class TokenStep:
    window_ids: tuple[int, ...]
    tokens: int


def group_windows(windows: list[PackedWindow], target_tokens: int) -> list[TokenStep]:
    if target_tokens < 1:
        raise ValueError("target_tokens_per_step 必须为正数")

    steps: list[TokenStep] = []
    bucket: list[int] = []
    used = 0
    for index, window in enumerate(windows):
        if bucket and used + window.tokens > target_tokens:
            steps.append(TokenStep(tuple(bucket), used))
            bucket = []
            used = 0
        bucket.append(index)
        used += window.tokens
    if bucket:
        steps.append(TokenStep(tuple(bucket), used))
    return steps
