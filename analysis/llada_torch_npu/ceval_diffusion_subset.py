"""
分批评估：每个科目取前 N 道题（默认 5 道），按分组运行扩散解码，输出单科准确率与分组宏平均。
用于小样本功能检查，不据此推断 CEval 全量质量。

运行示例（容器内）：
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/workspace:/workspace/xllm-src/build/lib.linux-aarch64-cpython-311:$PYTHONPATH
python -m analysis.llada_torch_npu.ceval_diffusion_subset \
  --model /workspace/models/LLaDA-MoE-7B-A1B-Instruct \
  --device npu:0 \
  --per_subject 5 \
  --steps 24 --gen_length 32 --block_length 8
"""

from __future__ import annotations

import argparse
import sys
import torch
from analysis.llada_torch_npu.runtime import resolve_device, seed_execution

from analysis.llada_torch_npu.ceval_diffusion import build_prompt, extract_choice, diffusion_generate

SUBJECT_GROUPS = [
    ["accountant", "advanced_mathematics", "art_studies", "basic_medicine", "business_administration"],
    ["chinese_language_and_literature", "civil_servant", "clinical_medicine", "college_chemistry", "college_economics"],
    ["college_physics", "college_programming", "computer_architecture", "computer_network", "discrete_mathematics"],
    ["education_science", "electrical_engineer", "environmental_impact_assessment_engineer", "fire_engineer", "high_school_biology"],
    ["high_school_chemistry", "high_school_chinese", "high_school_geography", "high_school_history", "high_school_mathematics"],
    ["high_school_physics", "high_school_politics", "ideological_and_moral_cultivation", "law", "legal_professional"],
    ["logic", "mao_zedong_thought", "marxism", "metrology_engineer", "middle_school_biology"],
    ["middle_school_chemistry", "middle_school_geography", "middle_school_history", "middle_school_mathematics", "middle_school_physics"],
    ["middle_school_politics", "modern_chinese_history", "operating_system", "physician", "plant_protection"],
    ["probability_and_statistics", "professional_tour_guide", "sports_science", "tax_accountant", "teacher_qualification"],
    ["urban_and_rural_planner", "veterinary_medicine"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 模型路径")
    ap.add_argument("--revision", default=None, help="Model revision for remote weights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="npu:0", help="设备")
    ap.add_argument("--dataset_dir", default=None, help="Optional CEval cache directory")
    ap.add_argument("--per_subject", type=int, default=5, help="每科抽取题目数量")
    ap.add_argument("--groups", type=int, default=-1, help="仅跑前 N 组（默认 -1 表示全部组）")
    ap.add_argument("--group_start", type=int, default=1, help="从第几组开始（1-based）")
    ap.add_argument("--steps", type=int, default=24, help="扩散步数")
    ap.add_argument("--gen_length", type=int, default=32, help="生成长度")
    ap.add_argument("--block_length", type=int, default=8, help="block 长度")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--cfg_scale", type=float, default=0.0)
    ap.add_argument("--logits_chunk", type=int, default=0, help="分块 logits 大小（>0 启用）")
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
    args = ap.parse_args()

    device = resolve_device(args.device)
    from transformers import AutoModel, AutoTokenizer
    seed_execution(args.seed, device)
    dtype = torch.float16 if device.type == "npu" else torch.float32

    print(f"[INFO] loading model on {device}, dtype={dtype}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, revision=args.revision)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, revision=args.revision, torch_dtype=dtype).to(device).eval()

    from datasets import load_dataset
    group_macros = []
    start_idx = max(args.group_start - 1, 0)
    groups_to_run = SUBJECT_GROUPS[start_idx:]
    if args.groups > 0:
        groups_to_run = groups_to_run[: args.groups]
    for gi, subjects in enumerate(groups_to_run):
        real_group_idx = start_idx + gi + 1
        accs = {}
        print(f"\n=== Group {real_group_idx}/{len(SUBJECT_GROUPS)} ===")
        for subject in subjects:
            ds = load_dataset(
                "ceval/ceval-exam",
                subject,
                split="val",
                cache_dir=args.dataset_dir,
                trust_remote_code=True,
            ).select(range(args.per_subject))
            correct = 0
            total = len(ds)
            for item in ds:
                prompt = build_prompt(item)
                label = item["answer"].strip().upper()
                ids = torch.tensor(tok(prompt)["input_ids"], device=device).unsqueeze(0)
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
                text = tok.batch_decode(
                    out_ids[:, ids.shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                pred = extract_choice(text)
                correct += int(pred == label)
            acc = correct / total if total > 0 else 0.0
            accs[subject] = acc
            print(f"{subject}: acc={acc:.3f} ({correct}/{total})")
        macro = sum(accs.values()) / len(accs)
        group_macros.append(macro)
        print(f"Group {real_group_idx} macro: {macro:.4f}")

    if group_macros:
        overall_macro = sum(group_macros) / len(group_macros)
        print(
            f"\nOverall macro (group-averaged, {args.per_subject} questions/subject, groups run={len(group_macros)}): {overall_macro:.4f}"
        )
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
