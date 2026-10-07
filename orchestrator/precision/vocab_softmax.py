"""按文字裁词表后的交叉熵。

打开之后，lm_head 只保留中文、英文和中英文符号（逗号、句号、引号这类）。
法语音符、俄文、日文假名、韩文等其他文字的行不参与 softmax，梯度是 0。
lm_head 的矩阵行数变成保留下来的 token 数，这一步的计算量按保留比例下降。
模型主体的注意力和 MLP 不变。

纯 ASCII 的法语词（bon、jour）和英文共用字母，分不开，会留下来。
带音符的法语（é、è、à、ç）会丢掉。
"""

from __future__ import annotations

import torch


class VocabRestriction:
    def __init__(self, kept_ids: torch.Tensor, lookup: torch.Tensor):
        self.kept_ids = kept_ids.contiguous()
        self.lookup = lookup

    @property
    def size(self) -> int:
        return int(self.kept_ids.numel())

    def project(self, weight: torch.Tensor) -> torch.Tensor:
        """取出保留行。反向把梯度散回原来的行，其余行梯度为 0。"""
        return _GatherRows.apply(weight, self.kept_ids)

    def remap(self, labels: torch.Tensor) -> torch.Tensor:
        lookup = self.lookup.to(device=labels.device)
        ignore = labels.eq(-100)
        mapped = lookup[labels.clamp(min=0)]
        missing = ~ignore & mapped.lt(0)
        if bool(missing.any()):
            raise RuntimeError("标签里有已裁掉的 token（中文、英文和符号以外）")
        return mapped.masked_fill(ignore, -100)


def restriction_from_ids(kept_ids: torch.Tensor, vocab_size: int) -> VocabRestriction:
    kept = torch.unique(kept_ids.to(dtype=torch.long, device="cpu"))
    if kept.numel() == 0:
        raise RuntimeError("词表裁剪后没有剩下任何 token")
    if int(kept[0]) < 0 or int(kept[-1]) >= vocab_size:
        raise RuntimeError(f"token id 超出词表大小 {vocab_size}")
    lookup = torch.full((vocab_size,), -1, dtype=torch.long)
    lookup[kept] = torch.arange(kept.numel())
    return VocabRestriction(kept, lookup)


def piece_text_kept(text: str) -> bool:
    """这段 token 文本是否整段都由中文、英文或中英文符号组成。"""
    if not text or "\ufffd" in text:
        return False
    return all(_char_allowed(char) for char in text)


def crop_zh_en_vocab(tokenizer, vocab_size: int) -> VocabRestriction:
    """用 tokenizer 把每个 id 解出来，只留下中文、英文、符号，以及对话和图像占位。"""
    special_ids = {int(token_id) for token_id in getattr(tokenizer, "all_special_ids", [])}
    limit = min(vocab_size, len(tokenizer))
    kept = []
    for token_id in range(limit):
        # 特殊 token 不看文字内容。裁掉 <|image_pad|> 这类占位后，模板对不上。
        if token_id in special_ids or piece_text_kept(_decode_piece(tokenizer, token_id)):
            kept.append(token_id)
    if not kept:
        raise RuntimeError("词表裁剪后没有剩下任何 token")
    return restriction_from_ids(torch.tensor(kept, dtype=torch.long), vocab_size)


def _decode_piece(tokenizer, token_id: int) -> str:
    # convert_ids_to_tokens 看到的是字节映射后的乱码，要还原成真正的字符再判断。
    piece = tokenizer.convert_ids_to_tokens(token_id)
    if piece is None:
        return ""
    return tokenizer.convert_tokens_to_string([piece])


def _char_allowed(char: str) -> bool:
    code = ord(char)
    # 英文、数字、英文符号，以及空格和换行。
    if char in "\t\n\r" or 0x20 <= code <= 0x7E:
        return True
    # 汉字，含扩展 A 和兼容汉字。
    if 0x3400 <= code <= 0x4DBF or 0x4E00 <= code <= 0x9FFF or 0xF900 <= code <= 0xFAFF:
        return True
    # 罕见汉字扩展区。
    if 0x20000 <= code <= 0x2EBEF or 0x2F800 <= code <= 0x2FA1F or 0x30000 <= code <= 0x3134F:
        return True
    # 中文标点：、。《》「」等。
    if 0x3000 <= code <= 0x303F:
        return True
    # 中英文排版常用符号：间隔号、破折号、弯引号、省略号。
    if code in {0x00B7, 0x2013, 0x2014, 0x2018, 0x2019, 0x201C, 0x201D, 0x2026}:
        return True
    # 全角英文、数字和全角逗号句号。半角片假名不在这里。
    if 0xFF01 <= code <= 0xFF5E or 0xFFE0 <= code <= 0xFFE6:
        return True
    return False


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, kept_ids: torch.Tensor) -> torch.Tensor:
        index = kept_ids.to(device=weight.device)
        ctx.save_for_backward(index)
        ctx.vocab_size = weight.shape[0]
        return weight.index_select(0, index)

    @staticmethod
    def backward(ctx, grad_rows: torch.Tensor):
        if grad_rows is None:
            return None, None
        (index,) = ctx.saved_tensors
        grad_weight = grad_rows.new_zeros(ctx.vocab_size, grad_rows.shape[1])
        grad_weight.index_add_(0, index, grad_rows)
        return grad_weight, None
