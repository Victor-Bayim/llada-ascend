# LLaDA-MoE-7B on Ascend 910B

[LLaDA](https://github.com/ML-GSAI/LLaDA) (LLaDA-MoE-7B-A1B-Instruct) running on Huawei Ascend 910B through [xLLM](https://github.com/jd-opensource/xllm).

The wall is the output projection. A full token×vocabulary tensor does not fit a single 910B once the sequence and the batch grow. The diffusion sampler now projects the vocabulary in chunks and the lm_head only gathers masked positions, so that tensor is never allocated. Cross-chunk argmax and confidence stay exact. Transfer selection and the tensor update in the sampling loop are vectorized.

`ceval_diffusion.py` is the sampler. `bench_diffusion.py` times generation only: sync before the run, sync after, then per-run stats and the flags that produced them. `tests/test_maintenance.py` covers full, chunked, and masked projections, block masks, zero masks, a synthetic full decode, and heterogeneous top-k. CEval smoke (first five subjects, two questions each) hits macro accuracy 0.7 on the quality setting.

`--logits_chunk`, `--masked_lm_head`, `--mm_bmm_nd` trade kernel launches against peak memory. `--device cpu` runs the same path off the NPU. Single-910B numbers (batch 1 and 16, length up to 2048, 2 diffusion steps) are in `data/reported-latency.csv`. Related xLLM C++ sources are under `archive/original/xllm-src/`.

```bash
python tools/validate_cpu.py
python -m pytest tests/test_maintenance.py

python -m analysis.llada_torch_npu.bench_diffusion \
  --model models/LLaDA-MoE-7B-A1B-Instruct \
  --device npu:0 \
  --batch_size 16 --prompt_len 2048 --gen_length 2048 \
  --logits_chunk 64 --masked_lm_head --mm_bmm_nd
```
