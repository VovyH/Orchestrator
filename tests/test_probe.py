"""探针能把 TinyCausalLM 的每一层和层内 aten 算子拆开。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 让同目录的 _bootstrap 可见
from _bootstrap import ensure_importable

ensure_importable()  # Orchestrator 是根项目还是子项目，都能 import orchestrator

import torch

from orchestrator.modeling import TinyCausalLM
from orchestrator.probe import profile_step


def test_probe_reports_layers_and_ops():
    torch.manual_seed(0)
    model = TinyCausalLM(vocab_size=32, hidden=16, layers=2, heads=4)
    model.train()
    input_ids = torch.randint(0, 32, (1, 12))
    labels = input_ids.clone()
    labels[0, 0] = -100
    cu = torch.tensor([0, 12], dtype=torch.int32)
    weight = model.lm_head.weight

    def step():
        hidden = model(input_ids, cu)
        logits = torch.nn.functional.linear(hidden, weight)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        loss.backward()
        return float(loss.detach())

    loss, report = profile_step(model, step)
    assert loss > 0
    assert "accelerator  cpu" in report
    assert "layers.0  forward" in report
    assert "layers.1  backward" in report
    assert "layers.0.attn  forward" in report
    assert "aten::" in report
    print(report)


if __name__ == "__main__":
    test_probe_reports_layers_and_ops()
    print("ok")
