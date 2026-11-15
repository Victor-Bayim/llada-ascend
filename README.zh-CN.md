[English](README.md) | **简体中文**

# LLaDA-MoE-7B 在昇腾 910B 上

[LLaDA](https://github.com/ML-GSAI/LLaDA)（LLaDA-MoE-7B-A1B-Instruct）通过 [xLLM](https://github.com/jd-opensource/xllm) 跑在华为昇腾 910B 上。

卡点在输出投影。序列和 batch 一拉长，完整的 token×词汇表张量在单卡 910B 上放不下。扩散采样器改成词汇表分块投影，lm_head 只收被掩码的位置，这张张量不再分配。跨块 argmax 和置信度仍是精确值。采样循环里的传输选择和张量更新做了向量化。

`ceval_diffusion.py` 是采样器。`bench_diffusion.py` 只计生成：跑之前同步，跑之后同步，再写每一轮和当时的标志。`tests/test_maintenance.py` 覆盖完整、分块、掩码三条投影路径，块掩码，零掩码，合成完整解码，异构 top-k。CEval 冒烟（前五个学科，每科两题）在质量设置下宏准确率 0.7。

`--logits_chunk`、`--masked_lm_head`、`--mm_bmm_nd` 在内核启动次数和峰值内存之间换。`--device cpu` 走同一条路径，不上 NPU。单卡 910B 的数字（batch 1 和 16，长度到 2048，2 步扩散）在 `data/reported-latency.csv`。相关 xLLM C++ 源码在 `archive/original/xllm-src/`。

```bash
python tools/validate_cpu.py
python -m pytest tests/test_maintenance.py

python -m analysis.llada_torch_npu.bench_diffusion \
  --model models/LLaDA-MoE-7B-A1B-Instruct \
  --device npu:0 \
  --batch_size 16 --prompt_len 2048 --gen_length 2048 \
  --logits_chunk 64 --masked_lm_head --mm_bmm_nd
```
