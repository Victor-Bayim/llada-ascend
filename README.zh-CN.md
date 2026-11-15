[English](README.md) | **简体中文**

# 昇腾上的 LLaDA-MoE：扩散推理与内存路径实验

本仓库包含 Victor 华为 ICT 项目的 Python 扩散推理实现、CPU 回归测试、benchmark/CEval 入口和报告转录结果。
Victor 担任队长，独立实现该赛题的适配。模型、采样方法和推理框架保留上游归属。

## 从实现开始

- [ceval_diffusion.py](analysis/llada_torch_npu/ceval_diffusion.py)：词表分块投影、精确跨块 argmax/置信度、掩码位置 lm_head、向量化转移选择和张量更新。
- [bench_diffusion.py](analysis/llada_torch_npu/bench_diffusion.py)：仅测生成阶段，包含显式设备检查、同步、逐轮计时和元数据。
- [CPU 回归测试](tests/test_maintenance.py)：完整/分块/掩码投影、块掩码、零掩码、合成完整解码和异构 top-k。
- [原始快照](archive/README.zh-CN.md)：原样保留 Python 与四份 xLLM C++ 上下文文件，SHA-256 见 [SOURCE-MANIFEST.json](SOURCE-MANIFEST.json)。
- [维护说明](maintenance/README.zh-CN.md)：原始行为与可运行修复。

词表分块避免完整构造 token×词表张量。Masked lm_head 只收集被掩码的隐藏向量再做输出投影；Transformer 主干仍处理完整序列。
这是内存路径优化，不是整个 Transformer 的稀疏执行。除非显式启用近似开关，置信度始终跨所有词表块精确计算。

## CPU 检查：无需模型、数据集或加速器

需要 Python 3.10+。在仓库根目录创建虚拟环境，再安装适用的 CPU torch wheel 与 pytest：

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python tools/verify_records.py
python tests/verify_cpu.py
python -m pytest -q tests/test_maintenance.py
# Or run the complete CPU validation and regenerate its scoped JSON record:
python tools/validate_cpu.py
```

记录校验器仅使用标准库。`verify_cpu.py` 保留针对归档函数的原始 39 项检查，确认原始 top-k 反例并验证维护修复，输出 `validation/current-cpu-checks.json`。
pytest 导入维护模块，在不依赖 Transformers、datasets、torch_npu 或权重的情况下检查更多路径。

## 模型与昇腾环境

上游项目：

- [LLaDA](https://github.com/ML-GSAI/LLaDA)：原始扩散语言模型工作。
- [LLaDA-MoE-7B-A1B-Instruct 模型卡](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct)：源码脚本使用的模型系列。
- [xLLM](https://github.com/jd-opensource/xllm)：归档 C++ 实验的框架背景。

单独获取完整官方模型快照，包括权重、配置、tokenizer 和自定义 Python 模型文件。安装 Hugging Face CLI 后执行：

```sh
# Set MODEL_REVISION to a commit selected from the official model's history.
hf download inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision "$MODEL_REVISION" --local-dir models/LLaDA-MoE-7B-A1B-Instruct
```

对于远程模型 ID，Python 入口也接受 `--revision`。对于本地模型目录，新测量应一并保留下载 commit 和文件哈希。
入口通过 Transformers 加载官方自定义模型代码；复现实验时请检查并保留所选模型快照及其环境。
仓库不包含权重、数据集和加速器运行时。

使用 Linux，以及厂商支持的相互兼容的 Ascend 910B 驱动/CANN、torch 和 torch_npu 组合。
[requirements-inference.txt](requirements-inference.txt) 列出入口直接导入的依赖。
历史容器标签为 `anoosxj/huaweiict-llada-xllm:0.7.1-slim`。

## 维护版运行入口

所有命令从仓库根目录执行。默认 `npu:0` 在扩展或设备不存在时明确失败；设备初始化和同步错误会向上传递。
`--device cpu` 表示有意执行 CPU 功能运行，并在 benchmark 输出中单独标记。CPU 模式拒绝 NPU 调优开关；设备检查完成前不加载模型。

与报告中 batch=16、length=2048 的形状和开关一致的长序列示例：

```sh
python -m analysis.llada_torch_npu.bench_diffusion \
  --model models/LLaDA-MoE-7B-A1B-Instruct --device npu:0 \
  --batch_size 16 --prompt_len 2048 --gen_length 2048 \
  --block_length 1024 --steps 2 --temperature 0 --threshold 0.95 \
  --warmup 1 --iters 5 --seed 0 --logits_chunk 64 --masked_lm_head \
  --inference_mode --mm_bmm_nd --json_out results/optimized.json
```

维护版 benchmark 使用 perf_counter，每次测量前后同步，排除模型加载时间，并输出每轮结果、mean/median/min/std、实际设备/dtype、模型 revision 参数、随机种子、开关及 Python 包版本。
CANN/驱动版本需另行记录。`--iters 1` 与历史次数一致，但不足以反映稳定的延迟分布。
完整历史形状与开关组合见 [run_perf.sh](analysis/final_report/run_perf.sh)，默认测量五轮。使用该 shell 包装脚本前，将 `MODEL_DIR` 设置为本地模型目录。

新的 A/B 测量应使用相同模型、环境、随机种子、形状和解码设置，将不带 `--masked_lm_head` 的 `--logits_chunk 0` 与分块和掩码变体对比；分别记录 OOM、延迟和加速器显存。
这些改动可能以额外的 kernel 启动开销换取更低峰值显存。

小规模质量烟测（前五个学科，各两题）：

```sh
python -m analysis.llada_torch_npu.ceval_diffusion_subset \
  --model models/LLaDA-MoE-7B-A1B-Instruct --device npu:0 \
  --per_subject 2 --groups 1 --group_start 1 --steps 24 \
  --gen_length 32 --block_length 8 --temperature 0.7 --seed 0
```

数据来自 `ceval/ceval-exam` 的 val 划分，可用 `--dataset_dir` 指定缓存。
`ceval_diffusion.py` 提供全学科顺序评测入口；CLI `llada_diffusion_cli.py` 接受 `--prompt` 和单个 `--device`（别名 `--devices`）。
共享批量采样器仅支持长度相等且兼容的 prompt。

## 历史测量与范围

[data/reported-latency.csv](data/reported-latency.csv) 包含技术报告转录的 14 行记录，明确标记为转录而非原始执行日志。
项目平台有八张 Ascend 910B；**这些测量使用单卡**。

| 设置 | 延迟报告 | 质量烟测 |
|---|---|---|
| 输入 | 重复 BOS，输入/输出长度 32–2048 | CEval 前 5 学科 × 2 题 |
| Batch | 1 和 16 | 每次一题 |
| 扩散步数 | 2 | 24 |
| 温度 | 0 | 0.7 |
| 重复次数 | 每个形状 1 次预热 + 1 次计时 | 10 题 |
| 报告质量 | 延迟配置下未测量 | 宏平均准确率 0.7，使用单独配置 |

[data/experiment.json](data/experiment.json) 保留上述范围。原始 batch=1 计时启用了向量化 top-k；batch=16 未启用。

## Python 扩散与 xLLM C++ 是两条独立路径

维护版性能/质量入口直接调用 Transformers AutoModel，不导入归档 C++ 模型或 xLLM 服务运行时。
`archive/original/xllm-src/` 下的四份 C++ 文件只提供实现上下文，不是完整框架检出。
实验性自回归 `ceval_eval.py` 同样归档保存，不是扩散 benchmark。

## 归属与范围

参见 [NOTICE.md](NOTICE.zh-CN.md)。源码注释与上游许可声明均保留。
模型权重、CEval 数据和加速器运行时遵循各自上游条款。
