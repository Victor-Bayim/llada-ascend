[English](README.md) | **简体中文**

# LLaDA-MoE-7B 在昇腾 910B 上的内存路径优化扩散推理

独立将 LLaDA-MoE-7B 的扩散推理适配至华为昇腾 910B。通过词汇表分块投影与 masked lm_head，避免完整 token×vocabulary 张量的 materialize，从而在内存受限条件下支持更长序列与更大 batch。

## 工程亮点

- 词汇表分块投影，跨 chunk 精确计算 argmax 与 confidence
- Masked lm_head：仅收集活跃隐藏状态后再进行最终投影
- 向量化的传输选择与张量更新
- 完整的 CPU 回归测试套件 + 同步的 NPU 基准测试框架
- 受控质量冒烟测试（CEval 子集，宏准确率 0.7）

## 支持的工作负载

- Batch size 1 与 16，提示长度最高 2048
- 显式设备同步以保证计时准确
- 标志控制的权衡（`--logits_chunk`、`--masked_lm_head`、`--mm_bmm_nd`）
- 完整的 CPU 验证路径（功能检查无需 NPU）

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
