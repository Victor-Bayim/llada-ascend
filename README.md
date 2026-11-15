**English** | [简体中文](README.zh-CN.md)

# LLaDA-MoE-7B on Ascend 910B: Memory-Path Optimized Diffusion Inference

Independent adaptation of LLaDA-MoE-7B diffusion inference to Huawei Ascend 910B. Vocabulary chunking and a masked lm_head eliminate full token×vocabulary materialization, enabling longer sequences and larger batches under tight NPU memory constraints.

## What Was Implemented

The core engineering addresses the memory bottleneck of diffusion-language-model inference:

- Vocabulary-chunked projection that computes logits in tiles while still producing exact cross-chunk argmax and confidence scores.
- Masked lm_head that gathers only the active (masked) hidden states before the final output projection, avoiding allocation of the full sequence×vocabulary tensor.
- Vectorized transfer selection and in-place tensor updates for the diffusion sampling loop.
- A complete CPU regression suite that validates full, chunked, and masked projection paths, block masks, zero masks, and heterogeneous top-k behavior.
- A synchronized NPU benchmark harness that records per-run timings with explicit device checks and metadata.
- A controlled quality smoke test on a CEval subset (macro accuracy 0.7 under the reported settings).

Flag-controlled options (`--logits_chunk`, `--masked_lm_head`, `--mm_bmm_nd`) allow trading kernel-launch overhead for lower peak memory.

## Supported Workloads & Results

- Batch sizes 1 and 16, prompt / generation lengths up to 2048.
- Explicit device synchronization for accurate latency measurement.
- Full CPU validation path (no NPU required for functional checks).
- Historical latency numbers (transcribed from the technical report) are preserved in `data/reported-latency.csv` for the single-device Ascend 910B runs.

## Quick Start

```bash
# CPU validation
python tools/validate_cpu.py
python -m pytest tests/test_maintenance.py

# NPU benchmark example
python -m analysis.llada_torch_npu.bench_diffusion \
  --model models/LLaDA-MoE-7B-A1B-Instruct \
  --device npu:0 \
  --batch_size 16 --prompt_len 2048 --gen_length 2048 \
  --logits_chunk 64 --masked_lm_head --mm_bmm_nd
```

Model weights and CEval data are not bundled. Upstream: LLaDA, LLaDA-MoE-7B-A1B-Instruct, xLLM (context only).

Independent implementation. Upstream model, sampling algorithm, and frameworks retain original authorship.
