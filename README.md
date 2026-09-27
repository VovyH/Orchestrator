# Orchestrator

**An Adaptive Sequence-Layout Scheduler for Memory-Efficient Fine-Tuning**

单卡训练调度器。模型仍是 PyTorch / HuggingFace，Orchestrator 只决定：**这一步怎么组数据、词表 loss 怎么算、用什么精度和优化器**。不写 CUDA kernel。

和 LLaMA-Factory 这类完整 SFT 工厂不同：那边管模板、数据集、DeepSpeed、多卡、LoRA 配方；这边把 packing、分块词表、精度、fused Adam、compile 做成可独立开关、可单独测量的调度项。packing 不划算时退回并排 batch，类似 AdaLoRA 的 try-off。

---

## 实测效果

硬件：本地 RTX 4080 16GB。模型：Qwen3-VL-2B（bf16 + 梯度检查点）。数据：Stage2 多帧。每组 **10 步**（稳态取第 2–10 步）。非 fused 组用 8-bit Adam（2B 的 FP32 Adam 状态单独约 16GB，这张卡放不下）。图像上限 262144 像素，序列上限 4096。注意力：`flash_attention_2`。

| 配置 | 条/步 | token/步 | 步时 | 吞吐 | PyTorch 峰值 |
|---|---:|---:|---:|---:|---:|
| baseline（1 条、整段 logits） | 1 | 1979 | 0.87 s | 2264 tok/s | 13901 MiB |
| chunked_loss（分块词表） | 1 | 1979 | 0.78 s | 2541 tok/s | 13205 MiB |
| packing（桶内拼接 + varlen，整段 logits） | 2 | 3058 | 3.64 s | 841 tok/s | 14652 MiB |
| **smart（桶内 varlen packing + 分块 loss）** | 2 | 3058 | 0.87 s | **3508 tok/s** | **13458 MiB** |

smart 相对 baseline：**吞吐 1.55×**，一步墙钟几乎相同，token 多约 55%，显存峰值更低。loss 从 0.65 降到 0.44。

要点：

- 分块词表 loss 少存整段 `[token, vocab]` logits，长步更明显。Qwen3.5-0.8B 上词表裁剪（只留中英文和符号）约保留 73% 词表，训练侧 lm_head 大约快 6–9%；解码墙钟几乎不变。
- packing 必须走 **长度分桶 + flash varlen**，不能把长短样本拼成一条再喂 `[S,S]` 稠密 mask（那会把 sdpa 打回 math 核，吞吐会掉到 baseline 以下）。
- 拼长之后仍算整段 logits，长块会被词表打穿。packed 步在 `chunk_tokens=0` 时会自动改成 2048。
- 这张 16GB 卡上，当前 FP8 实现前向走 `torch._scaled_mm`、反向是 FP32 GEMM，比 bf16 慢；fused FP32 Adam 会溢出到共享内存。A100 80GB 上 fused 相对普通 Adam 大约快 6%，那是两边状态都装得下的情况。

---

## 和 LLaMA-Factory 等框架的区别

| | Orchestrator | LLaMA-Factory / 同类 SFT 工厂 |
|---|---|---|
| 角色 | 单卡**调度器**：组 batch、loss、精度、优化器 | 完整训练工厂：数据、模板、LoRA、DeepSpeed、多卡、UI |
| Packing | 按长度分桶；`auto` 并排，`on` 只在桶内拼接并走 varlen flash；开训可试跑，慢就退回 batch | 通常拼接成长序列；padding 收益和注意力块长绑在一起，长短混在同一步会抬高 block 调度成本 |
| 词表 | 分块交叉熵，反向按块重算 logits；可裁到中英文+符号 | 默认整段 lm_head + CE |
| 精度 / 优化器 | FP8、fused Adam、`torch.compile` 探测后自动降级 | 跟 DeepSpeed / bitsandbytes / 配置模板走，不是按步可消融的独立开关 |
| 测量 | Stage2 消融把 packing / FP8 / fused / 分块 loss 拆开，带利用率、显存、tok/s | 主路径是「一种配方训完」，消融要自己改配置重跑 |
| 不负责 | 多卡、数据集清洗、对话模板库、RLHF | 这些是 LF 的主场 |

一句话：要配好 SFT 流水线用 LLaMA-Factory；要在**同一份权重**上把 packing 和词表 loss 等开关拆开测、并按长度分布自动组步，用 Orchestrator。

---

## 入口

| 入口 | 做什么 |
|---|---|
| `python -m orchestrator.cli --config ...` | 主训练循环 |
| `python -m orchestrator.run_stage2_ablation` | Stage2 多模态消融（packing / FP8 / fused / 分块 loss） |
| `python tests/test_orchestrator.py` | CPU 单测（packing、分块 loss、试跑还原权重） |

常用参数：

```text
python -m orchestrator.cli --config orchestrator/configs/smoke.yaml
python -m orchestrator.cli --config orchestrator/configs/qwen35_0_8b.yaml --packing auto
python -m orchestrator.cli --config orchestrator/configs/qwen35_0_8b.yaml --crop-vocab
```

`--packing`：`auto` / `on` / `off`。  
`--crop-vocab`：交叉熵只在中文、英文和中英文符号上 softmax。

Stage2 消融用环境变量指模型与数据（不要把机器路径写进配置仓库）：

```text
export ORCH_MODEL=/path/to/Qwen3-VL-2B
export ORCH_DATA=/path/to/stage2.json
export ORCH_IMAGES=/path/to/images
export ORCH_OUT=./runs
export ABLATION_STEPS=10
export ACCELERATOR=gpu
python -m orchestrator.run_stage2_ablation
```

---

## 运行方法

依赖：Python 3.10+、PyTorch。GPU 训练需要对应 CUDA 的 torch；VLM 消融需要 `transformers`、`pillow`。packing 走 varlen 需要 `flash-attn`。16GB 上 2B 全参建议再装 `bitsandbytes` 做 8-bit Adam。

```bash
git clone https://github.com/VovyH/Orchestrator.git
cd Orchestrator
pip install -e .
# 可选：pip install transformers pillow bitsandbytes flash-attn
```

CPU 冒烟（合成变长序列，不下载模型）：

```bash
python -m orchestrator.cli --config orchestrator/configs/smoke.yaml
python tests/test_orchestrator.py
python tests/test_vocab_softmax.py
python tests/test_backend.py
python tests/test_probe.py
```

HuggingFace 因果模型（改 yaml 里的 `model_name_or_path` 和 `dataset_path`）：

```bash
python -m orchestrator.cli --config orchestrator/configs/qwen35_0_8b.yaml --packing auto
```

`dataset_path` 是 jsonl：每行 `{"input_ids": [...]}`，或在能加载 tokenizer 时用 `{"text": "..."}`。不填则用配置里的合成长度。

---

## 调度规则（组数据）

1. 丢掉长于 `max_seq_len` 的样本。  
2. 按长度排序，桶内最长 / 最短 ≤ `bucket_ratio`（默认 1.25），长短不进同一步。  
3. `max_token = min(pack_length_limit, 2 × 全数据 p90)`。一步条数 = `max_token / 该桶 p90`。  
4. `packing: auto|off`：桶内**并排**，块长停在桶内最长，不把相近长度接成一条更长的块。  
5. `packing: on`：同一套分桶和条数，桶内拼接，交给 flash varlen；拼起来超过 `max_seq_len` 就拆回一条一步。  
6. `packing: auto` 且 `packing_probe_steps > 0`：若计划里还有拼接步，开训前用同一组样本比 packing 和 batch 的 tok/s，batch 不慢就把拼接步拆回 batch，并还原权重和优化器状态。

---

## 代码结构

```text
Orchestrator/
├── orchestrator/
│   ├── cli.py                 # 命令行入口
│   ├── config.py              # TrainConfig；yaml 未知字段直接报错
│   ├── trainer.py             # 主循环：组步、前向、分块 loss、试跑 packing
│   ├── backend.py             # gpu / ppu / cpu 运行时、计时、峰值显存
│   ├── modeling.py            # TinyCausalLM，CPU 验证 packing；CUDA 优先 flash varlen
│   ├── metrics.py             # 一步 tok/s、waste、可选 MFU
│   ├── monitor.py             # 后台线程采 nvidia-smi / ppu-smi
│   ├── probe.py               # 指定 step 的逐层 / 逐算子耗时
│   ├── stage2_experiment.py   # VLM Stage2 编码、消融、packing 自动分块 loss
│   ├── run_stage2_ablation.py # Stage2 消融启动器
│   ├── configs/
│   │   ├── smoke.yaml         # CPU 冒烟
│   │   └── qwen35_0_8b.yaml   # 0.8B 示例（改模型路径）
│   ├── data/
│   │   ├── adaptive.py        # 分桶、max token、计划一步 packed 还是并排
│   │   ├── packing.py         # 按长度双指针装窗（仍给旧路径和单测用）
│   │   ├── sampler.py         # 按 token 预算把窗口收成 step
│   │   ├── collator.py        # 拼接（cu_seqlens）或并排 pad
│   │   └── dataset.py         # jsonl 或合成变长序列
│   ├── precision/
│   │   ├── chunked_loss.py    # 分块交叉熵；-100 不进 lm_head
│   │   ├── vocab_softmax.py   # 裁到中英文和符号
│   │   ├── policy.py          # FP8 探测与 Linear 替换；反向保持高精度
│   │   └── checkpoint.py      # 隔若干层 activation checkpoint
│   └── compute/
│       ├── attention.py       # 选 flash_attention_2；packing 传 cu_seqlens、跳过 4D mask
│       ├── optimizer.py       # PPU FusedAdamW → torch fused AdamW → 普通 AdamW
│       └── compile_blocks.py  # 只 compile mlp 和注意力，避免变长把整图打爆
└── tests/                     # CPU 单测，不要求 GPU
```

---

## 配置字段（`TrainConfig`）

| 字段 | 含义 |
|---|---|
| `packing` | `auto` / `on` / `off` |
| `bucket_ratio` | 桶内最长不超过最短的倍数 |
| `pack_length_limit` | 由长度分布算出的 max token 上限 |
| `packing_probe_steps` | `auto` 时开训试跑步数 |
| `chunk_tokens` | 词表分块；`0` 表示一次算完（packed 步仍会自动改成 2048） |
| `restrict_vocab` / `--crop-vocab` | 只留中英文和符号 |
| `precision` | `bf16` / `fp8` / `auto` |
| `fused_adam` | 能 fused 就 fused |
| `compile_mlp` / `compile_attention` | 是否 `torch.compile` 对应模块 |
| `checkpoint_every_n` | 隔几层做 activation checkpoint |
| `device` | `auto` / `cpu` / `gpu` / `cuda` / `ppu` |

---

## 许可

MIT
