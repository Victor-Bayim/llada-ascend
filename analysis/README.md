# Runnable diffusion path

- `llada_torch_npu/ceval_diffusion.py`: shared maintained sampler and CEval evaluator.
- `llada_torch_npu/bench_diffusion.py`: synchronized timing and JSON metadata.
- `llada_torch_npu/ceval_diffusion_subset.py`: small quality smoke test.
- `llada_torch_npu/llada_diffusion_cli.py`: interactive prompt entry.
- `llada_torch_npu/runtime.py`: explicit device checks; no silent NPU-to-CPU fallback.
- `final_report/`: maintained shape-matrix and smoke-test shell wrappers.

Use `python -m analysis.llada_torch_npu.<entry>` from the repository root. Model weights,
HF inference dependencies and a compatible accelerator environment are required for
real inference. CPU tensor tests need no model or HF dependencies. See the root README.
Original source, including experimental xLLM serving, stays in `archive/original/`.
