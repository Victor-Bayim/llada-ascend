# LLaDA-MoE on Ascend: diffusion inference and memory-path experiments

This repository contains the actual Python diffusion implementation, CPU regressions,
benchmark/CEval entry points and report-transcribed results for Victor's Huawei ICT
project. Victor was team captain and independently implemented the adaptation for
this problem. The model, sampling approach and inference frameworks retain their
upstream authorship.

## Start with the implementation

- [ceval_diffusion.py](analysis/llada_torch_npu/ceval_diffusion.py): vocabulary-chunked
  projection, exact cross-chunk argmax/confidence, masked-position lm_head,
  vectorized transfer selection and tensor updates.
- [bench_diffusion.py](analysis/llada_torch_npu/bench_diffusion.py): generation-only
  timing with explicit device checks, synchronization, per-run timings and metadata.
- [CPU regressions](tests/test_maintenance.py): full/chunked/masked projections,
  block masks, zero masks, synthetic complete decoding and heterogeneous top-k.
- [Original snapshots](archive/README.md): unchanged Python and four xLLM C++ context
  files, with SHA-256 in [SOURCE-MANIFEST.json](SOURCE-MANIFEST.json).
- [Maintenance notes](maintenance/README.md): original behavior versus runnable fixes.

Vocabulary chunking avoids materializing the full token-by-vocabulary tensor.
Masked lm_head gathers only masked hidden vectors before output projection; the
Transformer backbone still processes the full sequence. This is a memory-path
optimization, not sparse execution of the whole Transformer. Exact confidence is
computed across all vocabulary chunks unless the explicit approximation flag is used.

## CPU checks: no model, dataset or accelerator

Python 3.10+ is required. From the repository root, create a virtual environment,
then install an appropriate CPU torch wheel and pytest:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-test.txt
python tools/verify_records.py
python tests/verify_cpu.py
python -m pytest -q tests/test_maintenance.py
# Or run the complete CPU validation and regenerate its scoped JSON record:
python tools/validate_cpu.py
```

The record verifier uses only the standard library. `verify_cpu.py` retains the
original 39-check suite against the archived functions, confirms the original top-k
counterexample and verifies the maintained correction; it writes
`validation/current-cpu-checks.json`. The pytest suite imports the maintained module
and checks additional paths without Transformers, datasets, torch_npu or weights.

## Model and Ascend environment

Upstreams:

- [LLaDA](https://github.com/ML-GSAI/LLaDA), the original diffusion-language-model work.
- [LLaDA-MoE-7B-A1B-Instruct model card](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct),
  the model family named by the source scripts.
- [xLLM](https://github.com/jd-opensource/xllm), context for the archived C++ experiments.

Obtain the complete official model snapshot (weights, config, tokenizer and custom
Python model files) separately. With the Hugging Face CLI installed:

```sh
# Set MODEL_REVISION to a commit selected from the official model's history.
hf download inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --revision "$MODEL_REVISION" --local-dir models/LLaDA-MoE-7B-A1B-Instruct
```

For remote model IDs the Python runners also accept `--revision`. For local model
folders, retain the downloaded commit and file hashes alongside any new measurements.
The runners load official custom model code through Transformers; inspect/retain the
selected model snapshot together with its environment when reproducing a run.
Weights, datasets and accelerator runtimes are not committed here.

Use Linux and a compatible Ascend 910B driver/CANN, torch and torch_npu combination
from the vendor's supported stack. [requirements-inference.txt](requirements-inference.txt)
lists direct runner imports. The historical
container tag was `anoosxj/huaweiict-llada-xllm:0.7.1-slim`.

## Maintained runners

All commands run from the repository root. Default `npu:0` fails explicitly if the
extension/device is absent; device initialization and synchronization errors propagate.
`--device cpu` is an intentional functional run and is labelled separately in benchmark
output. NPU tuning flags are rejected on CPU. No model is loaded before the device check.

A long-sequence example matching the report's batch=16, length=2048 shape/flags:

```sh
python -m analysis.llada_torch_npu.bench_diffusion \
  --model models/LLaDA-MoE-7B-A1B-Instruct --device npu:0 \
  --batch_size 16 --prompt_len 2048 --gen_length 2048 \
  --block_length 1024 --steps 2 --temperature 0 --threshold 0.95 \
  --warmup 1 --iters 5 --seed 0 --logits_chunk 64 --masked_lm_head \
  --inference_mode --mm_bmm_nd --json_out results/optimized.json
```

The maintained
benchmark uses perf_counter with synchronization before/after each measurement, excludes
model loading, and emits every run, mean/median/min/std, actual device/dtype, model
revision argument, seed, flags and Python package versions. Record CANN/driver versions
separately. `--iters 1` matches the historical count but is insufficient for a robust
latency distribution. The full historical shape/flag schedule is in
[run_perf.sh](analysis/final_report/run_perf.sh), with five measured runs by default.
Set `MODEL_DIR` to the local model folder before using that shell wrapper.

For a new A/B measurement, use the same model/environment/seed/shapes/decoding settings
and compare `--logits_chunk 0` without `--masked_lm_head` to chunked and masked variants;
record OOM outcomes, latency and accelerator memory separately. These changes can trade
kernel-launch overhead for lower peak memory.

A small quality smoke test (first five subjects, two questions each):

```sh
python -m analysis.llada_torch_npu.ceval_diffusion_subset \
  --model models/LLaDA-MoE-7B-A1B-Instruct --device npu:0 \
  --per_subject 2 --groups 1 --group_start 1 --steps 24 \
  --gen_length 32 --block_length 8 --temperature 0.7 --seed 0
```

The dataset is loaded from `ceval/ceval-exam`, val split; an optional `--dataset_dir`
selects the cache.
`ceval_diffusion.py` offers the full-subject sequential evaluation entry; the CLI
`llada_diffusion_cli.py` accepts `--prompt` and a single `--device` (`--devices` alias).
Only equal-length compatible prompts are supported by the shared batched sampler.

## Historical measurements and boundaries

[data/reported-latency.csv](data/reported-latency.csv) has 14 rows from a technical
report, explicitly marked as transcription rather than raw execution logs. The project
platform had eight Ascend 910B devices; **these measurements use one device**.

| Setting | Latency report | Quality smoke test |
|---|---|---|
| Input | repeated BOS, input/output length 32–2048 | CEval first 5 subjects × 2 questions |
| Batch | 1 and 16 | one question at a time |
| Diffusion steps | 2 | 24 |
| Temperature | 0 | 0.7 |
| Repetitions | 1 warm-up + 1 timed run per shape | 10 questions |
| Reported quality | not measured under the latency config | macro accuracy 0.7, separate config |

[data/experiment.json](data/experiment.json) retains these boundaries. The original batch=1 timings enabled
vectorized top-k; batch=16 did not.

## Python diffusion and xLLM C++ are separate paths

The maintained benchmark/quality runners call Transformers AutoModel directly.
They do not import the archived C++ model or xLLM serving runtime. The four C++ files
under `archive/original/xllm-src/` are implementation context only, not a complete
framework checkout. The experimental autoregressive `ceval_eval.py` is likewise
archived and is not the diffusion benchmark.

## Attribution and scope

See [NOTICE.md](NOTICE.md). Source comments and upstream license notices are retained.
Model weights, CEval data and accelerator runtimes follow their own upstream terms.
