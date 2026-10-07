# Orchestrator

**An Adaptive Sequence-Layout Scheduler for Memory-Efficient Fine-Tuning**

单卡微调时，墙钟和显存主要卡在两件事：**这一步的序列怎么排**，以及 **lm_head 要不要一次铺开整张词表**。Orchestrator 管的就是这两件事。模型仍是 PyTorch / HuggingFace，不写 CUDA kernel。

它有用，是因为同一张卡、同一份权重，一步能多算 token、峰值 logits 更小。RTX 4080 16GB 上 Qwen3-VL-2B 测到 **3508 tok/s，相对普通一步一条的 2264 tok/s 为 1.55×**，步时几乎不变，显存峰值更低。LLaMA-Factory 把数据、模板、LoRA、多卡配成能跑的流水线；它默认的 packing / 整段交叉熵不会给出这个倍数——按 LF 常见写法把样本拼长、再算整段 logits，在同一套测量里会掉到 841 tok/s，比不 packing 更慢。

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

差别不在「功能多少」，在**一步训练花多少时间和显存**。LF 解决的是「这套 SFT 怎么配齐」；Orchestrator 解决的是「配齐之后，单卡每秒能过多少 token」。缺模板、缺 LoRA、缺 DeepSpeed 不等于没用不——2B–7B 全参、一张消费卡，瓶颈就是步吞吐和 logits 峰值，不是缺一个 WebUI。

| | Orchestrator | LLaMA-Factory / 同类 SFT 工厂 |
|---|---|---|
| 要解决的问题 | 单卡一步的 token 吞吐和显存 | 把数据、模板、LoRA、DeepSpeed、多卡配成能训完的流水线 |
| Packing | 长度分桶，长短不进同一步；`on` 只在桶内拼接并走 flash varlen；开训试跑，更慢就退回并排。实测：只 packing 不算分块 loss → 841 tok/s | 常见路径是拼成长序列或 pad 到 batch 最长。padding「省」在空隙上，注意力仍按最长块计费；没有 varlen 时稠密 mask 会把 sdpa 打回 math 核 |
| 词表 / loss | 分块交叉熵，反向按块重算；可裁到中英文+符号。packing 必须和这一项一起开，否则拼长会被词表打穿 | 默认整段 `hidden @ vocab` 一次出齐 logits。词表越大、拼得越长，这一项越主导步时 |
| 实测（同卡同权重） | smart：3508 tok/s，1.55× baseline，显存更低 | 同等「拼长 + 整段 CE」落在 packing 那一行：比 baseline 慢 |
| 精度 / 优化器 | 探测后能 fused / FP8 / compile 再用，装不下或更慢就降级 | 跟 DeepSpeed、bitsandbytes、配置模板走，服务的是「这种配方能不能训」，不是「这一步 tok/s」 |
| 不做 | 对话模板库、数据集清洗、多卡 ZeRO、RLHF | 这些是 LF 已经做好的 |

独立开关是为了量清楚哪一项真的快。有用的是量完之后留下的组合：**分桶 + varlen packing + 分块词表**，不是「比 LF 少几个菜单」。

模板和多卡继续用 LF；单卡全参要把一步 token 做上去，用 Orchestrator 的组步和 loss。两者叠的是不同层，不是缩水版和完整版。

---

## 入口

| 入口 | 做什么 |
|---|---|
| `python run.py --model ... --data ... --out ...` | 一键微调，不要求 cwd 在项目根目录 |
| `python -m orchestrator.api --model ... --data ... --out ...` | 同上（模块方式调用） |
| `python -m orchestrator.cli --config ...` | 主训练循环（yaml 配置） |
| `python tests/test_orchestrator.py` | CPU 单测（packing、分块 loss、试跑还原权重） |
| `python tests/test_b_features.py` | CPU 单测（shuffle / 早停 / 断点续训） |

完整示例：`examples/train_qwen05b.py`（训练）、`examples/eval_qwen05b.py`（生成式评测）、
`examples/plot_or_metrics.py` / `plot_lf_metrics.py`（画 loss/grad 曲线）。

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

## 一键调用（fit）

不想碰 yaml 就用 `fit()`：给模型和数据，跑完存盘。

```python
from orchestrator import fit

result = fit(
    "/mnt/WYH/models/Qwen2.5-0.5B",
    dataset_path="/mnt/WYH/data/ev-charging-protocols/train.jsonl",
    output_dir="/mnt/WYH/runs/qwen05b",
    max_seq_len=1024,
    epochs=1,
    lr=1e-4,
)
print(result.format())
```

命令行等价：`python run.py --model ... --data ... --out ... --epochs 1`，不需要 cd 到项目根目录。  
完整例子：`examples/train_qwen05b.py`。

数据支持三种行格式（jsonl），或 txt 一行一条：

| 一行 | 怎么处理 |
|---|---|
| `{"text": "..."}` | tokenizer 编码，末尾补 eos |
| `{"input_ids": [...]}` | 直接用，适合预先分好词 |
| `{"messages": [{"role": ..., "content": ...}]}` | 走 tokenizer 的 chat template |

超长样本默认截断到 `max_seq_len`（`truncate_long=False` 改成丢弃）。

`fit()` 和原配置不一致、需要知道的几点：

- `packing` 默认 `"on"`。装了 flash-attn 就走 varlen 拼接，没装会自动退回并排。yaml 里的 `auto` 目前等于并排：`plan_steps` 只在 `mode == "on"` 时产出拼接步，`auto` 的开训试跑拿不到拼接步，因此不会触发。
- `epochs` 换算成 `max_steps`：先用调度器算出一轮多少步再乘。给了 `max_steps` 就以它为准。
- `batch_size` 只在并排时是「一步几条」。拼接时一步的 token 数是 `min(pack_length_limit, 2 × 全数据 p90)`，`pack_length_limit` 只是封顶。
- `precision` 默认 `bf16`。FP8 在这版实现里反向走 FP32 GEMM，README 实测更慢。
- 存盘用 `save_pretrained`，`output_dir` 下另有 `metrics.json` 记录每步 loss / tok/s / 峰值。

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
