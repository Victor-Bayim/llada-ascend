"""
LLaDA 扩散解码性能基准：统计延迟与吞吐量（tokens/s）。
适用于单卡 NPU 或 CPU（NPU 推荐设置 NPU_MEMORY_FRACTION）。

示例（容器内）：
export NPU_MEMORY_FRACTION=0.5
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:64
export PYTHONPATH=/workspace:/workspace/xllm-src/build/lib.linux-aarch64-cpython-311:$PYTHONPATH
python /workspace/analysis/llada_torch_npu/bench_diffusion.py \
  --model /workspace/models/LLaDA-MoE-7B-A1B-Instruct \
  --device npu:0 \
  --batch_size 1 \
  --prompt_len 256 \
  --gen_length 256 \
  --block_length 8 \
  --steps 24 \
  --iters 3 \
  --warmup 1
"""

from __future__ import annotations

import argparse
import os
import time

import torch
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None
from transformers import AutoModel, AutoTokenizer

from analysis.llada_torch_npu.ceval_diffusion import diffusion_generate


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型路径")
    ap.add_argument("--device", default="npu:0", help="设备")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--prompt_len", type=int, default=256)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=8)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--cfg_scale", type=float, default=0.0)
    ap.add_argument("--logits_chunk", type=int, default=0, help="分块 logits 大小（>0 启用）")
    ap.add_argument("--jit_compile", action="store_true", help="启用 NPU JIT 编译优化")
    ap.add_argument("--mm_bmm_nd", action="store_true", help="启用 NPU mm/bmm ND 格式")
    ap.add_argument("--inference_mode", action="store_true", help="使用 torch.inference_mode()")
    ap.add_argument("--masked_lm_head", action="store_true", help="仅对 mask 位置计算 lm_head")
    ap.add_argument("--masked_block_only", action="store_true", help="仅对当前 block 的 mask 位置计算 lm_head")
    ap.add_argument("--approx_confidence", action="store_true", help="使用 logit 近似置信度（跳过 softmax/logsumexp）")
    ap.add_argument("--gumbel_fp32", action="store_true", help="Gumbel 噪声使用 FP32")
    ap.add_argument("--block_topk_only", action="store_true", help="只对当前 block 做 topk")
    ap.add_argument("--vectorized_topk", action="store_true", help="批量 topk 选择（减少 Python 循环）")
    ap.add_argument("--use_where_update", action="store_true", help="使用 torch.where 更新，规避 index_put")
    ap.add_argument("--threshold", type=float, default=0.95, help="置信度阈值（0 表示关闭）")
    ap.add_argument("--prefix_look", type=int, default=0, help="block 前额外窗口（token 数）")
    ap.add_argument("--after_look", type=int, default=0, help="block 后额外窗口（token 数）")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    return ap.parse_args()


def sync_if_needed(device: torch.device):
    if hasattr(torch, "npu") and "npu" in str(device):
        try:
            torch.npu.synchronize()
        except Exception:
            pass


def main():
    args = parse_args()
    use_npu = args.device.startswith("npu") and hasattr(torch, "npu") and torch.npu.is_available()
    device = torch.device(args.device if use_npu else "cpu")
    dtype = torch.float16 if use_npu else torch.float32
    if use_npu and hasattr(torch, "npu"):
        try:
            torch.npu.set_device(device)
        except Exception:
            pass
        if args.jit_compile:
            try:
                torch.npu.set_compile_mode(True)
            except Exception:
                pass
        if args.mm_bmm_nd:
            try:
                torch.npu.set_mm_bmm_format_nd(True)
            except Exception:
                pass

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    if bos_id == 156895:
        bos_id = 1
    prompt_ids = torch.full(
        (args.batch_size, args.prompt_len),
        bos_id,
        dtype=torch.long,
        device=device,
    )

    context = torch.inference_mode if args.inference_mode else torch.no_grad
    with context():
        # warmup
        for _ in range(args.warmup):
            _ = diffusion_generate(
                model,
                prompt_ids,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                logits_chunk_size=args.logits_chunk,
                masked_lm_head=args.masked_lm_head,
                vectorized_topk=args.vectorized_topk,
                use_where_update=args.use_where_update,
                masked_block_only=args.masked_block_only,
                approx_confidence=args.approx_confidence,
                gumbel_fp32=args.gumbel_fp32,
                block_topk_only=args.block_topk_only,
                threshold=args.threshold,
                prefix_look=args.prefix_look,
                after_look=args.after_look,
            )
            sync_if_needed(device)

        times = []
        for _ in range(args.iters):
            t0 = time.time()
            _ = diffusion_generate(
                model,
                prompt_ids,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                logits_chunk_size=args.logits_chunk,
                masked_lm_head=args.masked_lm_head,
                vectorized_topk=args.vectorized_topk,
                use_where_update=args.use_where_update,
                masked_block_only=args.masked_block_only,
                approx_confidence=args.approx_confidence,
                gumbel_fp32=args.gumbel_fp32,
                block_topk_only=args.block_topk_only,
                threshold=args.threshold,
                prefix_look=args.prefix_look,
                after_look=args.after_look,
            )
            sync_if_needed(device)
            times.append(time.time() - t0)

    avg = sum(times) / len(times)
    tokens = args.batch_size * args.gen_length
    tps = tokens / avg if avg > 0 else 0.0
    print(f"device={device} batch={args.batch_size} prompt_len={args.prompt_len} gen_len={args.gen_length}")
    print(f"steps={args.steps} block_length={args.block_length} iters={args.iters} warmup={args.warmup}")
    print(f"threshold={args.threshold} prefix_look={args.prefix_look} after_look={args.after_look}")
    print(f"latency_avg_s={avg:.4f} throughput_tokens_per_s={tps:.2f}")
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
