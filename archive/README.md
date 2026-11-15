**English** | [简体中文](README.zh-CN.md)

# Original source context

`original/` preserves the 13 source files supplied in the competition-source evidence
collection, byte for byte. `../SOURCE-MANIFEST.json` records their SHA-256 hashes.

The original shell scripts name an xLLM `0.7.1-slim` container tag.

`original/xllm-src/` has FOUR context files: LLaDA registration/config handling,
Qwen3-MoE integration, a torch MoE fallback, and a single-process worker path.
This is **not a complete/buildable xLLM checkout or a validated serving release**.
Referenced headers and build configuration are absent. Do not attempt to compile it
as a standalone framework.

`original/analysis/llada_torch_npu/ceval_eval.py` is an experimental xLLM serving
evaluation path. It is archived, not the maintained diffusion/latency entry point.
The maintained Python runners use Transformers directly; running them inside an
xLLM container does not make the C++ serving path responsible for their results.

Original scripts contain historical CPU fallback, unsorted batched top-k and forced
exit behavior. Use `analysis/` for maintained execution.
