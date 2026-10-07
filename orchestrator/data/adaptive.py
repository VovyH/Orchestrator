"""按长度分布组 batch。

先把长度接近的样本放进同一个桶，长的和短的不会进同一步。
max token 用全数据第 90 百分位的两倍算出来，再封顶。
每个桶的块长是自己的第 90 百分位，一步放的条数 = max token / 块长。
并排时每条的块长停在自己那一档，不会被另一档的长度拖长。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlannedStep:
    sample_ids: tuple[int, ...] # ① 样本下标（索引）
    packed: bool # ② 拼不拼
    content_tokens: int # ③ 有效 token 数（总长度）
    slots: int # ④ 占位数（长度）


@dataclass(frozen=True)
class LayoutPlan:
    steps: list[PlannedStep]
    dropped: list[int]
    max_token: int
    buckets: int


def normalize_packing(value) -> str:
    if value is True or (isinstance(value, str) and value.lower() in {"on", "true", "pack", "1"}):
        return "on"
    if value is False or (isinstance(value, str) and value.lower() in {"off", "false", "batch", "0"}):
        return "off"
    if isinstance(value, str) and value.lower() in {"auto", "adaptive"}:
        return "auto"
    raise ValueError(f"packing 只能是 auto、on 或 off，收到 {value!r}")


def waste_ratio(lengths: list[int]) -> float:
    """同一组样本如果并排、垫到最长那条，多算出来的 token 比例。"""
    if len(lengths) <= 1:
        return 0.0
    slots = len(lengths) * max(lengths)
    if slots <= 0:
        return 0.0
    return 1.0 - sum(lengths) / slots


def prefer_pack(lengths: list[int], waste_min: float, pack_limit: int) -> bool:
    """空隙够大、接起来又没有超过块长上限，才保留 packing。"""
    if len(lengths) < 2:
        return False
    total = sum(lengths)
    if total > pack_limit:
        return False
    return waste_ratio(lengths) >= waste_min


def group_into_batches(sample_ids: list[int], lengths: list[int], batch_size: int) -> list[PlannedStep]:
    if batch_size < 1:
        raise ValueError("nopack_batch_size 必须为正数")
    ordered = sorted(sample_ids, key=lambda index: (lengths[index], index))
    steps: list[PlannedStep] = []
    for start in range(0, len(ordered), batch_size):
        group = tuple(ordered[start : start + batch_size])
        chosen = [lengths[index] for index in group]
        content = sum(chosen)
        slots = len(chosen) * max(chosen)
        steps.append(PlannedStep(group, False, content, slots))
    return steps


def percentile(values: list[int], quantile: float) -> int:
    """quantile 取 0 到 100。线性插值后四舍五入。"""
    if not values:
        raise ValueError("没有长度，没法算分位数")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (quantile / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return int(round(ordered[low] * (1.0 - weight) + ordered[high] * weight))


def bucket_ids(lengths: list[int], max_seq_len: int, ratio: float) -> tuple[list[list[int]], list[int]]:
    """按长度排序后切桶。桶内最长不超过最短的 ratio 倍。超长样本丢弃。"""
    if ratio < 1.0:
        raise ValueError("bucket_ratio 不能小于 1")
    # 丢弃超长样本，并记录
    dropped = [index for index, length in enumerate(lengths) if length > max_seq_len]
    kept = [index for index, length in enumerate(lengths) if 0 < length <= max_seq_len]
    # 排序
    kept.sort(key=lambda index: (lengths[index], index))
    buckets: list[list[int]] = []
    current: list[int] = []
    # 遍历数据，放入桶
    for index in kept:
        if not current:
            current = [index]
            continue
        shortest = lengths[current[0]]
        if shortest > 0 and lengths[index] / shortest <= ratio:
            current.append(index)
        else:
            buckets.append(current)
            current = [index]
    if current:
        buckets.append(current)
    return buckets, dropped # 返回的是下标和舍弃的数据


def max_token_from_lengths(lengths: list[int], ceiling: int) -> int:
    """全数据第 90 百分位的两倍。并排两条时块长仍是单条，这个数只决定一步放多少条。"""
    if ceiling < 1:
        raise ValueError("pack_length_limit 必须为正数")
    if not lengths:
        return ceiling
    base = max(percentile(lengths, 90), 1)
    return min(ceiling, base * 2)


def _steps_from_buckets(
    buckets: list[list[int]],
    lengths: list[int],
    budget: int,
    max_seq_len: int,
    packed: bool,
) -> list[PlannedStep]:
    """
    把每个桶切成实际的训练步。
    bucket_ids 决定「谁和谁一个桶」，这一步决定「每个桶里的样本怎么分组、一步装几条、拼不拼」
    ——是调度链的最后一环（④）。
    """
    steps: list[PlannedStep] = []
    # 切桶
    for bucket in buckets:
        chosen = [lengths[index] for index in bucket] # 遍历：桶内所有样本的长度
        block = max(percentile(chosen, 90), 1) # 桶里 90% 的样本都不超过这个长度。
        count = max(1, budget // block) # 整除，向下取整
        for start in range(0, len(bucket), count):
            group = tuple(bucket[start : start + count])
            group_lengths = [lengths[index] for index in group]
            content = sum(group_lengths)
            if packed:
                if content <= max_seq_len:
                    steps.append(PlannedStep(group, True, content, content))
                else:
                    for index, length in zip(group, group_lengths):
                        steps.append(PlannedStep((index,), True, length, length))
            else:
                # 同一档并排。接成一条会把块长变成总和，调度比垫到桶内最长更贵。
                slots = len(group_lengths) * max(group_lengths)
                steps.append(PlannedStep(group, False, content, slots))
    return steps


def plan_steps(
    lengths: list[int],
    max_seq_len: int,
    mode: str,
    batch_size: int,
    waste_min: float,
    pack_limit: int,
    bucket_ratio: float = 1.25,
) -> LayoutPlan:
    """
    训练前的「组步调度器」：输入所有样本的长度，输出「每一步该装哪几条样本、拼不拼接」的完整计划。
    它不碰 token，只做排列组合的规划——实际拼张量是后面 collator 的事。
    """
    buckets, dropped = bucket_ids(lengths, max_seq_len, bucket_ratio) # 把样本按长度进行排序，然后切分成桶
    kept_lengths = [lengths[index] for bucket in buckets for index in bucket]
    budget = max_token_from_lengths(kept_lengths, pack_limit)
    packed = mode == "on"
    steps = _steps_from_buckets(buckets, lengths, budget, max_seq_len, packed)
    return LayoutPlan(steps, dropped, budget, len(buckets))


def demote_packed(steps: list[PlannedStep], lengths: list[int], batch_size: int) -> list[PlannedStep]:
    """试跑判定 packing 不划算之后，把还标着 packed 的样本重新按 batch 分组。"""
    loose = [index for step in steps if step.packed for index in step.sample_ids]
    kept = [step for step in steps if not step.packed]
    return kept + group_into_batches(loose, lengths, batch_size)
