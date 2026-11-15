"""
CEval 精度评测脚本（单卡 torch_npu 回退，适配 xLLM 0.7.1）。
功能：
- 下载/读取 CEval 数据集（val/test/dev），默认 val。
- 组装多选题 Prompt，调用 xLLM 推理，解析 A/B/C/D 选项。
- 计算单科与宏平均准确率，输出到终端与日志。

使用示例（容器内）：
export XLLM_FORCE_TORCH_MOE=1
export XLLM_SINGLE_PROCESS=1
python3 analysis/llada_torch_npu/ceval_eval.py \
  --model /workspace/models/LLaDA-MoE-7B-A1B-Instruct \
  --devices npu:0 \
  --dataset_dir /workspace/datasets/ceval \
  --split val \
  --batch_size 8 \
  --log_file /workspace/logs/ceval_eval.log
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

from datasets import load_dataset
from xllm import ArgumentParser as XLLMArgParser
from xllm import LLM, RequestParams


SUBJECTS: List[str] = [
    "accountant",
    "advanced_mathematics",
    "art_studies",
    "basic_medicine",
    "business_administration",
    "chinese_language_and_literature",
    "civil_servant",
    "clinical_medicine",
    "college_chemistry",
    "college_economics",
    "college_physics",
    "college_programming",
    "computer_architecture",
    "computer_network",
    "discrete_mathematics",
    "education_science",
    "electrical_engineer",
    "environmental_impact_assessment_engineer",
    "fire_engineer",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_chinese",
    "high_school_geography",
    "high_school_history",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_politics",
    "ideological_and_moral_cultivation",
    "law",
    "legal_professional",
    "logic",
    "mao_zedong_thought",
    "marxism",
    "metrology_engineer",
    "middle_school_biology",
    "middle_school_chemistry",
    "middle_school_geography",
    "middle_school_history",
    "middle_school_mathematics",
    "middle_school_physics",
    "middle_school_politics",
    "modern_chinese_history",
    "operating_system",
    "physician",
    "plant_protection",
    "probability_and_statistics",
    "professional_tour_guide",
    "sports_science",
    "tax_accountant",
    "teacher_qualification",
    "urban_and_rural_planner",
    "veterinary_medicine",
]


def build_prompt(item: Dict[str, str]) -> str:
    """构造单题 Prompt，要求仅输出 A/B/C/D。"""
    question = item["question"].strip()
    opts = [item["A"].strip(), item["B"].strip(), item["C"].strip(), item["D"].strip()]
    demo = (
        "示例：\n"
        "题目：示例题目，正确选项是第三个。\n"
        "A. 选项一\nB. 选项二\nC. 选项三\nD. 选项四\n"
        "回答：C\n"
        "请注意：只输出一个大写字母，不要输出任何其他字符或空格。\n"
    )
    prompt = (
        "你是一名考试助手，请阅读题目和选项，判断唯一正确答案，"
        "务必只输出选项字母（A/B/C/D），不要输出句子或解释。\n"
        + demo +
        f"题目：{question}\n"
        f"A. {opts[0]}\n"
        f"B. {opts[1]}\n"
        f"C. {opts[2]}\n"
        f"D. {opts[3]}\n"
        "请直接回答一个字母（A/B/C/D），不要输出其它字符。"
    )
    return prompt


def extract_choice(text: str) -> str:
    """从模型输出中提取首个 A/B/C/D。找不到则返回空串。"""
    if not text:
        return ""
    m = re.search(r"[ABCD]", text.upper())
    return m.group(0) if m else ""


def run_eval(args) -> Tuple[float, Dict[str, float]]:
    # 初始化 xLLM
    xparser = XLLMArgParser()
    # 仅使用我们关心的参数；其余保持默认
    xllm_args = xparser.parser.parse_args(
        [
            "--model",
            args.model,
            "--devices",
            args.devices,
            "--communication_backend",
            "none",
            "--max_memory_utilization",
            str(args.max_mem_util),
        ]
    )

    llm = LLM(**vars(xllm_args))
    rp = RequestParams()
    rp.max_tokens = args.max_tokens
    rp.temperature = 0.0
    rp.top_p = 1.0

    subject_acc: Dict[str, float] = {}

    for subject in SUBJECTS:
        ds = load_dataset(
            "ceval/ceval-exam",
            subject,
            split=args.split,
            cache_dir=args.dataset_dir,
            trust_remote_code=True,
        )
        prompts = [build_prompt(item) for item in ds]
        labels = [item["answer"].strip().upper() for item in ds]

        correct = 0
        total = len(prompts)

        # 批量推理
        for i in range(0, total, args.batch_size):
            batch_prompts = prompts[i : i + args.batch_size]
            batch_labels = labels[i : i + args.batch_size]
            outputs = llm.generate(batch_prompts, rp, True)
            for out, label in zip(outputs, batch_labels):
                text = out.outputs[0].text
                pred = extract_choice(text)
                if pred == label:
                    correct += 1

        acc = correct / total if total > 0 else 0.0
        subject_acc[subject] = acc
        print(f"[{subject}] acc={acc:.4f} ({correct}/{total})")
        sys.stdout.flush()

    macro_acc = sum(subject_acc.values()) / len(subject_acc)
    return macro_acc, subject_acc


def main():
    parser = argparse.ArgumentParser(description="CEval 精度评测")
    parser.add_argument("--model", required=True, help="模型路径（HF 格式）")
    parser.add_argument("--devices", default="npu:0", help="设备列表，默认 npu:0")
    parser.add_argument("--dataset_dir", default="/workspace/datasets/ceval", help="CEval 缓存目录/HF_HOME")
    parser.add_argument("--split", default="val", choices=["dev", "val", "test"], help="评测 split")
    parser.add_argument("--batch_size", type=int, default=8, help="生成时的 batch 大小")
    parser.add_argument("--max_tokens", type=int, default=4, help="生成最大 token 数（越小越快）")
    parser.add_argument("--max_mem_util", type=float, default=0.85, help="xLLM 内存利用率参数")
    parser.add_argument("--log_file", default="", help="日志输出路径（可选）")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", args.dataset_dir)

    # 简单日志重定向
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        log_f = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = log_f
        sys.stderr = log_f

    macro_acc, subject_acc = run_eval(args)
    print("\n=== CEval 评测结果 ===")
    print(f"宏平均准确率: {macro_acc:.4f}")
    for k, v in sorted(subject_acc.items()):
        print(f"{k}\t{v:.4f}")
    # 避免底层 BlockManager 在解释器退出时做严格检查导致进程 abort，
    # 在结果输出后直接强制退出。
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
