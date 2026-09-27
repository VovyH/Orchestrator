"""packing、分块 loss 和两步训练的检查。不碰 GPU。"""

import torch
import torch.nn.functional as F

from orchestrator.config import TrainConfig
from orchestrator.data.adaptive import max_token_from_lengths, plan_steps, prefer_pack
from orchestrator.data.collator import collate_plan, collate_step
from orchestrator.data.packing import pack_by_length
from orchestrator.data.sampler import TokenStep, group_windows
from orchestrator.modeling import TinyCausalLM
from orchestrator.precision.chunked_loss import chunked_causal_lm_loss
from orchestrator.precision.policy import probe_native_fp8
from orchestrator.trainer import train


def test_two_pointer_fills_long_with_short():
    windows, dropped = pack_by_length([100, 100, 100, 500, 500], 600)
    assert dropped == []
    assert sorted(window.tokens for window in windows) == [100, 600, 600]
    seen = [sample_id for window in windows for sample_id in window.sample_ids]
    assert sorted(seen) == [0, 1, 2, 3, 4]
    for window in windows:
        assert window.tokens <= 600


def test_oversize_sample_is_dropped():
    windows, dropped = pack_by_length([1000, 10], 128)
    assert dropped == [0]
    assert len(windows) == 1
    assert windows[0].sample_ids == (1,)


def test_token_budget_groups_windows():
    windows, _ = pack_by_length([100, 100, 50], 100)
    steps = group_windows(windows, 150)
    assert [step.tokens for step in steps] == [100, 150]


def test_collator_resets_positions_and_blocks_boundary():
    sequences = [[1, 2, 3], [4, 5]]
    windows, _ = pack_by_length([3, 2], 8)
    batch = collate_step(sequences, windows, TokenStep((0,), 5), 8)
    assert batch.position_ids.tolist() == [[0, 1, 2, 0, 1]]
    assert batch.labels[0, 3].item() == -100
    assert batch.cu_seqlens.tolist() == [0, 3, 5]


def test_packed_segment_matches_solo_forward():
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=32, hidden=16, layers=2, heads=4).eval()
    sequences = [[1, 2, 3, 4], [5, 6, 7]]
    windows, _ = pack_by_length([4, 3], 8)
    packed = collate_step(sequences, windows, TokenStep((0,), 7), 8)
    solo_windows, _ = pack_by_length([3], 8)
    solo = collate_step([sequences[1]], solo_windows, TokenStep((0,), 3), 8)
    with torch.no_grad():
        packed_hidden = model(packed.input_ids, packed.cu_seqlens, packed.position_ids)
        solo_hidden = model(solo.input_ids, solo.cu_seqlens, solo.position_ids)
    assert torch.allclose(packed_hidden[0, 4:], solo_hidden[0], atol=1e-5)


def test_chunked_loss_matches_full_cross_entropy():
    torch.manual_seed(1)
    hidden = torch.randn(1, 21, 16, requires_grad=True)
    weight = torch.randn(32, 16, requires_grad=True)
    labels = torch.randint(0, 32, (1, 21))
    labels[0, 0] = -100
    labels[0, 7] = -100
    loss = chunked_causal_lm_loss(hidden, weight, labels, chunk_tokens=8)
    loss.backward()

    shift_hidden = hidden.detach().clone().requires_grad_(True)
    shift_weight = weight.detach().clone().requires_grad_(True)
    flat_hidden = shift_hidden[:, :-1, :].reshape(-1, 16)
    flat_labels = labels[:, 1:].reshape(-1)
    valid = flat_labels.ne(-100)
    reference = F.cross_entropy(F.linear(flat_hidden[valid], shift_weight), flat_labels[valid])
    reference.backward()

    assert torch.allclose(loss, reference, atol=1e-5)
    assert torch.allclose(hidden.grad, shift_hidden.grad, atol=1e-4)
    assert torch.allclose(weight.grad, shift_weight.grad, atol=1e-4)


def test_uniform_lengths_fall_back_to_batch():
    lengths = [100, 102, 98, 101]
    layout = plan_steps(lengths, max_seq_len=150, mode="auto", batch_size=2, waste_min=0.10, pack_limit=4096)
    assert layout.dropped == []
    assert layout.buckets == 1
    assert layout.steps
    assert all(not step.packed for step in layout.steps)
    assert all(len(step.sample_ids) == 2 for step in layout.steps)


def test_long_and_short_stay_in_different_buckets():
    lengths = [500, 40, 30, 20]
    assert prefer_pack(lengths, waste_min=0.10, pack_limit=4096)
    layout = plan_steps(lengths, max_seq_len=600, mode="auto", batch_size=2, waste_min=0.10, pack_limit=4096)
    assert layout.dropped == []
    assert layout.buckets == 4
    assert all(not step.packed for step in layout.steps)
    for step in layout.steps:
        assert 0 not in step.sample_ids or step.sample_ids == (0,)


def test_max_token_follows_the_length_distribution():
    lengths = [1000] * 10
    assert max_token_from_lengths(lengths, ceiling=4096) == 2000
    layout = plan_steps(lengths, max_seq_len=4096, mode="auto", batch_size=2, waste_min=0.10, pack_limit=4096)
    assert layout.max_token == 2000
    assert layout.buckets == 1
    assert all(len(step.sample_ids) == 2 for step in layout.steps)


def test_short_bucket_uses_the_dataset_budget():
    lengths = [200] * 4 + [2000] * 4
    layout = plan_steps(lengths, max_seq_len=4096, mode="auto", batch_size=2, waste_min=0.10, pack_limit=4096)
    assert layout.buckets == 2
    assert layout.max_token == 4000
    short_steps = [step for step in layout.steps if all(lengths[index] == 200 for index in step.sample_ids)]
    long_steps = [step for step in layout.steps if all(lengths[index] == 2000 for index in step.sample_ids)]
    assert short_steps and long_steps
    assert all(len(step.sample_ids) == 4 for step in short_steps)
    assert all(len(step.sample_ids) == 2 for step in long_steps)
    assert all(not step.packed for step in layout.steps)


def test_packing_on_keeps_long_and_short_apart():
    lengths = [200] * 4 + [2000] * 4
    layout = plan_steps(lengths, max_seq_len=4096, mode="on", batch_size=2, waste_min=0.10, pack_limit=4096)
    assert layout.buckets == 2
    assert all(step.packed for step in layout.steps)
    for step in layout.steps:
        group = [lengths[index] for index in step.sample_ids]
        assert len(set(group)) == 1
        assert sum(group) <= 4096
    short = [step for step in layout.steps if lengths[step.sample_ids[0]] == 200]
    long = [step for step in layout.steps if lengths[step.sample_ids[0]] == 2000]
    assert short and long
    assert all(len(step.sample_ids) == 4 for step in short)
    assert all(len(step.sample_ids) == 2 for step in long)


def test_padded_batch_matches_single_packed_row():
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=32, hidden=16, layers=2, heads=4).eval()
    sequences = [[1, 2, 3, 4]]
    from orchestrator.data.adaptive import PlannedStep

    packed = collate_plan(sequences, PlannedStep((0,), True, 4, 4))
    batched = collate_plan(sequences, PlannedStep((0,), False, 4, 4))
    with torch.no_grad():
        packed_hidden = model(packed.input_ids, packed.cu_seqlens, packed.position_ids)
        batched_hidden = model(batched.input_ids, None, batched.position_ids, padding_mask=batched.padding_mask)
    assert torch.allclose(packed_hidden, batched_hidden, atol=1e-5)


def test_cpu_smoke_train_finishes():
    meters = train(
        TrainConfig(
            device="cpu",
            max_steps=2,
            synthetic_samples=12,
            synthetic_min_len=4,
            synthetic_max_len=16,
            max_seq_len=32,
            target_tokens_per_step=48,
            chunk_tokens=8,
            checkpoint_every_n=2,
            tiny_hidden=16,
            tiny_layers=2,
            tiny_heads=4,
            vocab_size=32,
            log_every=0,
        )
    )
    assert len(meters) == 2
    assert all(torch.isfinite(torch.tensor(item.loss)) for item in meters)
    assert all(item.content_tokens > 0 for item in meters)
    assert all(0.0 <= item.padding_waste < 1.0 for item in meters)


def test_probe_restores_weights_before_training():
    from copy import deepcopy

    from orchestrator.backend import resolve_accelerator
    from orchestrator.compute.optimizer import create_optimizer
    from orchestrator.data.adaptive import PlannedStep
    from orchestrator.data.dataset import load_sequences
    from orchestrator.trainer import _build_model, _calibrate_packing

    cfg = TrainConfig(
        device="cpu",
        packing="auto",
        packing_probe_steps=1,
        max_seq_len=64,
        nopack_batch_size=2,
        pack_waste_min=0.10,
        pack_length_limit=4096,
        chunk_tokens=8,
        tiny_hidden=16,
        tiny_layers=1,
        tiny_heads=4,
        vocab_size=32,
        synthetic_samples=4,
        synthetic_min_len=4,
        synthetic_max_len=40,
        seed=1,
        log_every=0,
    )
    sequences = load_sequences(cfg)
    lengths = [len(sequence) for sequence in sequences]
    plans = [PlannedStep((0, 1), True, lengths[0] + lengths[1], lengths[0] + lengths[1])]
    accelerator = resolve_accelerator("cpu")
    model = _build_model(cfg, accelerator)
    optimizer = create_optimizer(model.parameters(), cfg.lr, cfg.betas, cfg.weight_decay, False)
    before = deepcopy(model.state_dict())
    _calibrate_packing(model, optimizer, sequences, plans, lengths, cfg, accelerator, 1, None, "auto")
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name])


def test_fp8_probe_stays_off_cpu():
    supported, reason = probe_native_fp8(torch.device("cpu"))
    assert supported is False
    assert reason


def test_varlen_kwargs_uses_int32_and_max_len():
    from orchestrator.compute.attention import varlen_kwargs

    cu = torch.tensor([0, 4, 10], dtype=torch.int64)
    kwargs = varlen_kwargs(cu)
    assert kwargs["cu_seq_lens_q"].dtype == torch.int32
    assert kwargs["max_length_q"] == 6
    assert kwargs["max_length_k"] == 6
    assert kwargs["is_causal"] is True


if __name__ == "__main__":
    test_two_pointer_fills_long_with_short()
    test_oversize_sample_is_dropped()
    test_token_budget_groups_windows()
    test_collator_resets_positions_and_blocks_boundary()
    test_packed_segment_matches_solo_forward()
    test_chunked_loss_matches_full_cross_entropy()
    test_uniform_lengths_fall_back_to_batch()
    test_long_and_short_stay_in_different_buckets()
    test_max_token_follows_the_length_distribution()
    test_short_bucket_uses_the_dataset_budget()
    test_packing_on_keeps_long_and_short_apart()
    test_padded_batch_matches_single_packed_row()
    test_cpu_smoke_train_finishes()
    test_probe_restores_weights_before_training()
    test_fp8_probe_stays_off_cpu()
    test_varlen_kwargs_uses_int32_and_max_len()
    print("ok")
