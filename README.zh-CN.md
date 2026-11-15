[English](README.md) | **简体中文**

# LLaDA-MoE-7B 在昇腾 910B 上的内存路径优化扩散推理

独立将 LLaDA-MoE-7B 的扩散推理适配至华为昇腾 910B。通过词汇表分块投影与 masked lm_head，避免完整 token×vocabulary 张量的 materialize，从而在内存受限条件下支持更长序列与更大 batch。

## 实现内容

核心工程解决扩散语言模型推理的内存瓶颈：

- 词汇表分块投影，在分块计算 logits 的同时仍产出精确的跨 chunk argmax 与 confidence。
- Masked lm_head 仅收集活跃（被掩码）的隐藏状态后再进行最终输出投影，避免分配完整序列×词汇表张量。
- 向量化的传输选择与就地张量更新，用于扩散采样循环。
- 完整的 CPU 回归测试套件，验证完整、分块与掩码投影路径、块掩码、零掩码以及异构 top-k 行为。
- 同步的 NPU 基准测试框架，记录带显式设备检查与元数据的逐轮计时。
- 受控质量冒烟测试（CEval 子集，报告设置下宏准确率 0.7）。

标志控制选项（`--logits_chunk`、`--masked_lm_head`、`--mm_bmm_nd`）允许在内核启动开销与更低峰值内存之间进行权衡。

## 支持的工作负载与结果

- Batch size 1 与 16，提示 / 生成长度最高 2048。
- 显式设备同步以保证延迟测量准确。
- 完整的 CPU 验证路径（功能检查无需 NPU）。
- 历史延迟数据（从技术报告转录）保存在 `data/reported-latency.csv`，对应单卡昇腾 910B 运行。

## 快速开始

```bash
# CPU 验证
python tools/validate_cpu.py
python -m pytest tests/test_maintenance.py

# NPU 基准测试示例
python -m analysis.llada_torch_npu.bench_diffusion \
  --model models/LLaDA-MoE-7B-A1B-Instruct \
  --device npu:0 \
  --batch_size 16 --prompt_len 2048 --gen_length 2048 \
  --logits_chunk 64 --masked_lm_head --mm_bmm_nd
```

模型权重与 CEval 数据未打包。上游：LLaDA、LLaDA-MoE-7B-A1B-Instruct、xLLM（仅作上下文参考）。

独立实现。上游模型、采样算法与框架保留原始作者归属。
