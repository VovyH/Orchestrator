"""分块交叉熵。

反向时按块重算 logits，图里不保留整段 [token, vocab] 的激活。
词表很大时，这一步决定 16k 序列会不会把显存打满。
"""

import torch
import torch.nn.functional as F


def _iter_logits(hidden: torch.Tensor, weight: torch.Tensor, chunk_tokens: int):
    total = hidden.shape[0]
    if chunk_tokens <= 0 or total <= chunk_tokens:
        yield F.linear(hidden, weight), 0, total
        return
    for start in range(0, total, chunk_tokens):
        end = min(start + chunk_tokens, total)
        yield F.linear(hidden[start:end], weight), start, end


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, chunk_tokens: int):
        ctx.save_for_backward(hidden, weight, labels)
        ctx.chunk_tokens = int(chunk_tokens)
        loss_sum = hidden.new_zeros((), dtype=torch.float32)
        with torch.no_grad():
            for logits, start, end in _iter_logits(hidden, weight, ctx.chunk_tokens):
                loss_sum = loss_sum + F.cross_entropy(logits.float(), labels[start:end], reduction="sum")
        return loss_sum / hidden.shape[0]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, labels = ctx.saved_tensors
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        scale = (grad_output / hidden.shape[0]).to(dtype=torch.float32)
        for logits, start, end in _iter_logits(hidden, weight, ctx.chunk_tokens):
            probs = torch.softmax(logits.float(), dim=-1)
            probs.scatter_add_(
                1,
                labels[start:end].unsqueeze(1),
                -torch.ones(end - start, 1, device=probs.device, dtype=probs.dtype),
            )
            dlogits = probs * scale
            grad_hidden[start:end].copy_((dlogits @ weight.float()).to(dtype=hidden.dtype))
            if grad_weight is not None:
                grad_weight.add_((dlogits.transpose(0, 1) @ hidden[start:end].float()).to(dtype=weight.dtype))
            del logits, probs, dlogits
        return grad_hidden, grad_weight, None, None


def chunked_causal_lm_loss(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    """hidden 的位置 t 预测 labels 的位置 t+1。-100 不进入 lm_head。"""
    shift_hidden = hidden[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    flat_hidden = shift_hidden.reshape(-1, shift_hidden.shape[-1])
    flat_labels = shift_labels.reshape(-1)
    valid = flat_labels.ne(-100)
    if not torch.any(valid):
        return hidden.sum() * 0.0
    return _ChunkedLinearCrossEntropy.apply(flat_hidden[valid], weight, flat_labels[valid], int(chunk_tokens))
