"""Generation-only benchmark of the maintained Transformers diffusion path.

Run from the repository root with python -m analysis.llada_torch_npu.bench_diffusion.
--device npu:N must resolve to a real NPU; --device cpu is an explicit functional run.
No xLLM C++ serving path is used here.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from analysis.llada_torch_npu.ceval_diffusion import diffusion_generate
from analysis.llada_torch_npu.runtime import (
    configure_npu, resolve_device, seed_execution, synchronize_device,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Local HF model directory or model ID")
    ap.add_argument("--revision", default=None, help="Pin remote model/custom-code commit")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--prompt_len", type=int, default=256)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--cfg_scale", type=float, default=0.0)
    ap.add_argument("--logits_chunk", type=int, default=0)
    for flag in ["jit_compile", "mm_bmm_nd", "inference_mode", "masked_lm_head",
                 "masked_block_only", "approx_confidence", "gumbel_fp32",
                 "block_topk_only", "vectorized_topk", "use_where_update"]:
        ap.add_argument("--" + flag, action="store_true")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--prefix_look", type=int, default=0)
    ap.add_argument("--after_look", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--json_out", type=Path, help="Save metadata and every measured run")
    args = ap.parse_args()
    if min(args.batch_size, args.prompt_len, args.gen_length, args.block_length, args.steps, args.iters) <= 0:
        ap.error("Shapes, steps and iters must be positive")
    if args.warmup < 0 or args.logits_chunk < 0:
        ap.error("warmup and logits_chunk must be non-negative")
    if args.gen_length % args.block_length or args.steps % (args.gen_length // args.block_length):
        ap.error("block_length must divide gen_length; number of blocks must divide steps")
    return args


def run_benchmark(args):
    # Resolve first, before imports/downloads of any external model code.
    device = resolve_device(args.device)
    configure_npu(device, jit_compile=args.jit_compile, mm_bmm_nd=args.mm_bmm_nd)
    seed_execution(args.seed, device)
    from transformers import AutoModel, AutoTokenizer
    dtype = torch.float16 if device.type == "npu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, revision=args.revision)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, revision=args.revision, torch_dtype=dtype,
    ).to(device).eval()
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    if bos_id == 156895:
        bos_id = 1
    prompt = torch.full((args.batch_size, args.prompt_len), bos_id, dtype=torch.long, device=device)
    kwargs = dict(
        steps=args.steps, gen_length=args.gen_length, block_length=args.block_length,
        temperature=args.temperature, cfg_scale=args.cfg_scale, logits_chunk_size=args.logits_chunk,
        masked_lm_head=args.masked_lm_head, vectorized_topk=args.vectorized_topk,
        use_where_update=args.use_where_update, masked_block_only=args.masked_block_only,
        approx_confidence=args.approx_confidence, gumbel_fp32=args.gumbel_fp32,
        block_topk_only=args.block_topk_only, threshold=args.threshold,
        prefix_look=args.prefix_look, after_look=args.after_look,
    )
    times = []
    context = torch.inference_mode if args.inference_mode else torch.no_grad
    with context():
        for _ in range(args.warmup):
            diffusion_generate(model, prompt, **kwargs)
            synchronize_device(device)
        for _ in range(args.iters):
            synchronize_device(device)
            start = time.perf_counter()
            diffusion_generate(model, prompt, **kwargs)
            synchronize_device(device)
            times.append(time.perf_counter() - start)
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for package in ["torch-npu", "transformers", "datasets"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    config = {key: value for key, value in vars(args).items() if key != "json_out"}
    result = {
        "device": str(device), "dtype": str(dtype), "versions": versions, "config": config,
        "execution_path": "Transformers diffusion; not xLLM C++ serving",
        "input": "repeated BOS tokens", "model_loading_included": False,
        "scope": "NPU generation latency" if device.type == "npu" else "explicit CPU functional run; not Ascend performance",
        "latencies_seconds": times, "latency_avg_s": statistics.mean(times),
        "latency_median_s": statistics.median(times), "latency_min_s": min(times),
        "latency_std_s": statistics.pstdev(times),
        "throughput_tokens_per_s": args.batch_size * args.gen_length / statistics.mean(times),
        "host_accelerator_stack": "Record CANN/driver versions separately; not detected by this runner",
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    result = run_benchmark(parse_args())
    print(json.dumps(result, indent=2))
    print(f"latency_avg_s={result['latency_avg_s']:.4f} throughput_tokens_per_s={result['throughput_tokens_per_s']:.2f}")


if __name__ == "__main__":
    main()
