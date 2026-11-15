[English](README.md) | **简体中文**

# LLaDA-MoE-7B 在昇腾 910B 上的内存路径优化扩散推理

将 LLaDA-MoE-7B 的扩散推理适配到昇腾 910B。主要改动在输出投影路径：词汇表按块处理， lm_head 只收集被掩码的位置，从而不再物化完整的 token×词汇表张量。这使得模型在单卡 910B 的内存限制下能跑更长序列和更大 batch。

`ceval_diffusion.py` 实现分块投影、跨块精确 argmax/置信度、masked lm_head 和向量化传输选择。`bench_diffusion.py` 只计时生成阶段，每次测量前后做设备同步，并输出逐轮统计和元数据。`tests/test_maintenance.py` 中的 CPU 回归覆盖完整/分块/掩码投影、块掩码、零掩码、合成完整解码和异构 top-k。小规模 CEval 冒烟测试（前五个学科各两题）在质量配置下报告宏准确率 0.7。

标志 `--logits_chunk`、`--masked_lm_head` 和 `--mm_bmm_nd` 控制内核启动开销与峰值内存之间的权衡。`--device cpu` 可在无 NPU 情况下跑同一套逻辑做功能检查。单卡 910B 上的历史延迟数据（batch 1 和 16，长度最高 2048，2 步扩散）转录在 `data/reported-latency.csv`。

仓库在 `archive/original/xllm-src/` 下保留了来自 xLLM 项目（京东开源）的四份 C++ 源文件。这些文件为早期在昇腾上用同一模型进行的 C++ 实验提供实现上下文；它们不被当前的 Python 扩散推理入口链接或使用，当前入口依赖 Transformers。xLLM 本身是一个侧重高效 LLM 服务的推理框架。

```bash
# CPU 检查
python tools/validate_cpu.py
python -m pytest tests/test_maintenance.py

# NPU 示例
python -m analysis.llada_torch_npu.bench_diffusion \
  --model models/LLaDA-MoE-7B-A1B-Instruct \
  --device npu:0 \
  --batch_size 16 --prompt_len 2048 --gen_length 2048 \
  --logits_chunk 64 --masked_lm_head --mm_bmm_nd
```

权重和 CEval 数据未包含。上游模型、采样方法和框架保留原始作者归属。
