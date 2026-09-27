"""裁剪后的交叉熵只在保留行上做 softmax；中英文符号留下，法语音符丢掉。"""

import torch
import torch.nn.functional as F

from orchestrator.config import TrainConfig
from orchestrator.precision.chunked_loss import chunked_causal_lm_loss
from orchestrator.precision.vocab_softmax import crop_zh_en_vocab, piece_text_kept, restriction_from_ids
from orchestrator.trainer import train


def test_piece_keeps_zh_en_symbols_and_drops_french_accents():
    assert piece_text_kept("hello")
    assert piece_text_kept("你好")
    assert piece_text_kept(",")
    assert piece_text_kept("，")
    assert piece_text_kept("。")
    assert piece_text_kept("bon")
    assert not piece_text_kept("é")
    assert not piece_text_kept("è")
    assert not piece_text_kept("пр")
    assert not piece_text_kept("안녕")
    assert not piece_text_kept("カ")


class _FakeTokenizer:
    def __init__(self, pieces: list[str], special_ids: list[int]):
        self.pieces = pieces
        self.all_special_ids = special_ids

    def __len__(self) -> int:
        return len(self.pieces)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.pieces[token_id]

    def convert_tokens_to_string(self, pieces: list[str]) -> str:
        return pieces[0]


def test_crop_keeps_special_tokens_and_zh_en():
    tokenizer = _FakeTokenizer(["hello", "é", "，", "привет", "。"], special_ids=[3])
    restriction = crop_zh_en_vocab(tokenizer, vocab_size=5)
    assert restriction.kept_ids.tolist() == [0, 2, 3, 4]


def test_restricted_loss_matches_subset_softmax():
    torch.manual_seed(0)
    hidden = torch.randn(1, 9, 8, requires_grad=True)
    weight = torch.randn(20, 8, requires_grad=True)
    labels = torch.tensor([[-100, 3, 3, 7, 1, -100, 7, 1, 3]])
    restriction = restriction_from_ids(torch.tensor([1, 3, 7]), 20)
    assert restriction.size == 3

    loss = chunked_causal_lm_loss(hidden, restriction.project(weight), restriction.remap(labels), chunk_tokens=4)

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    mapped = restriction.remap(labels)
    flat_hidden = hidden_ref[:, :-1, :].reshape(-1, 8)
    flat_labels = mapped[:, 1:].reshape(-1)
    valid = flat_labels.ne(-100)
    reference = F.cross_entropy(
        F.linear(flat_hidden[valid], weight_ref.index_select(0, restriction.kept_ids)),
        flat_labels[valid],
    )
    reference.backward()
    loss.backward()

    assert torch.allclose(loss, reference, atol=1e-5)
    assert torch.allclose(hidden.grad, hidden_ref.grad, atol=1e-4)
    assert torch.allclose(weight.grad, weight_ref.grad, atol=1e-4)
    dropped = torch.ones(20, dtype=torch.bool)
    dropped[restriction.kept_ids] = False
    assert torch.count_nonzero(weight.grad[dropped]) == 0
    assert torch.count_nonzero(weight.grad[restriction.kept_ids]) > 0


def test_dropped_language_label_is_rejected():
    restriction = restriction_from_ids(torch.tensor([1, 2]), 10)
    try:
        restriction.remap(torch.tensor([1, 5]))
    except RuntimeError:
        return
    raise AssertionError("被裁掉的 token 作为标签应该报错")


def test_tiny_model_refuses_crop():
    try:
        train(
            TrainConfig(
                device="cpu",
                max_steps=1,
                synthetic_samples=4,
                synthetic_min_len=4,
                synthetic_max_len=6,
                max_seq_len=16,
                target_tokens_per_step=16,
                chunk_tokens=8,
                tiny_hidden=16,
                tiny_layers=1,
                tiny_heads=4,
                vocab_size=32,
                log_every=0,
                restrict_vocab=True,
            )
        )
    except RuntimeError as exc:
        assert "crop-vocab" in str(exc)
        return
    raise AssertionError("没有 tokenizer 时应该拒绝裁剪")


if __name__ == "__main__":
    test_piece_keeps_zh_en_symbols_and_drops_french_accents()
    test_crop_keeps_special_tokens_and_zh_en()
    test_restricted_loss_matches_subset_softmax()
    test_dropped_language_label_is_rejected()
    test_tiny_model_refuses_crop()
    print("ok")
