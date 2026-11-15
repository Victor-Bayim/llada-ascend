**English** | [简体中文](README.zh-CN.md)

# LLaDA-MoE-7B on Ascend 910B: Memory-Path Optimized Diffusion Inference

Diffusion inference for LLaDA-MoE-7B adapted to Ascend 910B. The main change is in the output projection path: vocabulary is processed in chunks and the lm_head only gathers masked positions, so the full token×vocabulary tensor is never materialized. This lets the model run longer sequences and larger batches under the memory limits of a single 910B.

`ceval_diffusion.py` implements the chunked projection, exact cross-chunk argmax/confidence, masked lm_head, and vectorized transfer selection. `bench_diffusion.py` times only the generation phase with device synchronization before and after each measurement and writes per-run stats plus metadata. CPU regressions in `tests/test_maintenance.py` cover full/chunked/masked projections, block masks, zero masks, synthetic complete decoding, and heterogeneous top-k. A small CEval smoke test (first five subjects, two questions each) reports macro accuracy 0.7 under the quality configuration.

Flags `--logits_chunk`, `--masked_lm_head`, and `--mm_bmm_nd` control the trade-off between kernel-launch overhead and peak memory. `--device cpu` runs the same logic for functional checks without an NPU. Historical latency numbers from a single 910B (batch 1 and 16, lengths up to 2048, 2 diffusion steps) are transcribed in `data/reported-latency.csv`.

The repository also retains four C++ source files from the xLLM project (JD Open Source) under `archive/original/xllm-src/`. These files provide implementation context for earlier C++ experiments with the same model on Ascend; they are not linked or used by the current Python diffusion runners, which rely on Transformers. xLLM itself is an inference framework focused on efficient LLM serving.

```bash
# CPU checks
python tools/validate_cpu.py
python -m pytest tests/test_maintenance.py

# NPU example
python -m analysis.llada_torch_npu.bench_diffusion \
  --model models/LLaDA-MoE-7B-A1B-Instruct \
  --device npu:0 \
  --batch_size 16 --prompt_len 2048 --gen_length 2048 \
  --logits_chunk 64 --masked_lm_head --mm_bmm_nd
```

Weights and CEval data are not included. Upstream model, sampling method, and frameworks keep their original authorship.
