"""
LLaDA 扩散解码的简易 CLI，保持 xLLM 风格的参数（支持 --model / --devices），
但当模型类型为 LLaDA 时直接走官方扩散采样逻辑，而非 xLLM 自回归推理。

使用示例（容器内）：
source /usr/local/Ascend/ascend-toolkit/set_env.sh
PYTHONPATH=/workspace:/workspace/xllm-src/build/lib.linux-aarch64-cpython-311 \
python3 /workspace/analysis/llada_torch_npu/llada_diffusion_cli.py \
  --model /workspace/models/LLaDA-MoE-7B-A1B-Instruct \
  --devices npu:0 \
  --prompt "请简要介绍华为ICT大赛赛题4要做什么？" \
  --steps 12 --gen_length 32 --block_length 8
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

# 尝试兼容 xLLM 的 ArgumentParser 以接受设备配置；若不存在则降级自定义解析
try:
    from xllm import ArgumentParser as XLLMArgParser
except Exception:
    XLLMArgParser = None

from analysis.llada_torch_npu.ceval_diffusion import diffusion_generate


def parse_args():
    parser = argparse.ArgumentParser(description="LLaDA 扩散解码 CLI（xLLM 风格）")
    parser.add_argument("--model", required=True, help="HF 模型路径")
    parser.add_argument("--devices", default="npu:0", help="设备列表，默认 npu:0")
    parser.add_argument("--prompt", action="append", required=True, help="输入 prompt，可多次指定")
    parser.add_argument("--steps", type=int, default=12, help="扩散步数")
    parser.add_argument("--gen_length", type=int, default=32, help="生成 token 数（遮盖长度）")
    parser.add_argument("--block_length", type=int, default=8, help="每块 token 数，需整除 gen_length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Gumbel 温度")
    parser.add_argument("--cfg_scale", type=float, default=0.0, help="classifier-free guidance 系数")
    parser.add_argument("--logits_chunk", type=int, default=0, help="分块 logits 大小（>0 启用）")
    parser.add_argument("--masked_lm_head", action="store_true", help="仅对 mask 位置计算 lm_head")
    parser.add_argument("--masked_block_only", action="store_true", help="仅对当前 block 的 mask 位置计算 lm_head")
    parser.add_argument("--approx_confidence", action="store_true", help="使用 logit 近似置信度（跳过 softmax/logsumexp）")
    parser.add_argument("--gumbel_fp32", action="store_true", help="Gumbel 噪声使用 FP32")
    parser.add_argument("--block_topk_only", action="store_true", help="只对当前 block 做 topk")
    parser.add_argument("--vectorized_topk", action="store_true", help="批量 topk 选择（减少 Python 循环）")
    parser.add_argument("--use_where_update", action="store_true", help="使用 torch.where 更新，规避 index_put")
    parser.add_argument("--threshold", type=float, default=0.95, help="置信度阈值（0 表示关闭）")
    parser.add_argument("--prefix_look", type=int, default=0, help="block 前额外窗口（token 数）")
    parser.add_argument("--after_look", type=int, default=0, help="block 后额外窗口（token 数）")
    parser.add_argument("--hf_home", default="/workspace/datasets/ceval", help="HF 缓存目录")
    parser.add_argument("--log_file", default="", help="日志输出（可选）")

    # 如果 xLLM 的 ArgumentParser 可用，允许复用其设备/内存参数（兼容调用方式）
    if XLLMArgParser:
        xparser = XLLMArgParser()
        # 只解析设备/内存相关，忽略其余
        known, _ = xparser.parser.parse_known_args([])
        # 占位使用，实际设备由 --devices 控制
        _ = known

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ.setdefault("HF_HOME", args.hf_home)

    use_npu = (
        args.devices.startswith("npu")
        and hasattr(torch, "npu")
        and torch.npu.is_available()
    )
    device = torch.device(args.devices.split(",")[0] if use_npu else "cpu")
    dtype = torch.bfloat16 if use_npu else torch.float32

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        log_f = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = log_f
        sys.stderr = log_f

    print(f"[INFO] loading model: {args.model} on {device}, dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()

    prompts: List[str] = args.prompt
    for idx, p in enumerate(prompts):
        input_ids = tokenizer(p)["input_ids"]
        ids = torch.tensor(input_ids, device=device).unsqueeze(0)
        out_ids = diffusion_generate(
            model,
            ids,
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
        text = tokenizer.batch_decode(
            out_ids[:, ids.shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print(f"\n[Result {idx}]")
        print("Prompt:", p)
        print("Output:", text)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
