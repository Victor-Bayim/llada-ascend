**English** | [简体中文](README.zh-CN.md)

# LLaDA-MoE-7B on Ascend 910B: Memory-Path Optimized Diffusion Inference

Independent adaptation of LLaDA-MoE-7B diffusion inference to Huawei Ascend 910B. Vocabulary chunking and masked lm_head eliminate full token×vocabulary materialization, enabling longer sequences and larger batches under tight memory constraints.

## Engineering Highlights

- Vocabulary-chunked projection with exact cross-chunk argmax and confidence
- Masked lm_head that gathers only active hidden states before the final projection
- Vectorized transfer selection and tensor updates
- Complete CPU regression suite plus synchronized NPU benchmark harness
- Controlled quality smoke test (CEval subset, macro accuracy 0.7)

## Supported Workloads

- Batch sizes 1 and 16, prompt lengths up to 2048
- Explicit device synchronization for accurate timing
- Flag-controlled trade-offs (`--logits_chunk`, `--masked_lm_head`, `--mm_bmm_nd`)
- Full CPU validation path (no NPU required for functional checks)

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
