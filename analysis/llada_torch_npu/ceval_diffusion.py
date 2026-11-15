"""
基于 LLaDA 官方扩散采样逻辑的 CEval 精度评测（torch_npu，单卡）。
相比 xLLM 自回归路径，直接使用模型自带的扩散解码算法，避免因解码方式不匹配导致的乱输出。

用法示例（容器内）：
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m analysis.llada_torch_npu.ceval_diffusion \
  --model /workspace/models/LLaDA-MoE-7B-A1B-Instruct \
  --device npu:0 \
  --dataset_dir /workspace/datasets/ceval \
  --split val \
  --steps 24 \
  --gen_length 32 \
  --block_length 8 \
  --batch_size 1 \
  --log_file /workspace/logs/ceval_diffusion.log

说明：
- 默认 batch_size=1，避免扩散解码的 padding 处理复杂度；可以按需调大，但 prompt 需等长。
- steps 越大、gen_length 越长，输出质量可能提升，但耗时增加；需在精度/时延间权衡。
- 设备不可用时明确报错；CPU 功能运行必须显式指定 --device cpu。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple, TYPE_CHECKING

import torch
from analysis.llada_torch_npu.runtime import resolve_device, seed_execution

if TYPE_CHECKING:
    from transformers import AutoModel


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
        + demo
        + f"题目：{question}\n"
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


def add_gumbel_noise(
    logits: torch.Tensor, temperature: float, gumbel_fp32: bool = False
) -> torch.Tensor:
    if temperature == 0:
        return logits
    if gumbel_fp32:
        logits_fp32 = logits.float()
        noise = torch.rand_like(logits_fp32)
        gumbel_noise = (-torch.log(noise)).pow(temperature)
        return logits_fp32.exp() / gumbel_noise
    logits_fp64 = logits.to(torch.float64)
    noise = torch.rand_like(logits_fp64, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits_fp64.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True).to(torch.int64)
    base = mask_num // steps
    remainder = mask_num % steps
    step_idx = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    extra = (step_idx < remainder).to(torch.int64)
    return base + extra


def _build_transfer_index(
    confidence: torch.Tensor,
    num_transfer_tokens: torch.Tensor,
    step_idx: int,
    vectorized_topk: bool,
    out: torch.Tensor | None,
) -> torch.Tensor:
    """根据置信度构造本 step 的 transfer_index（bool）。"""
    device = confidence.device
    if out is None:
        transfer_index = torch.zeros_like(confidence, dtype=torch.bool, device=device)
    else:
        transfer_index = out
        transfer_index.fill_(False)
    if vectorized_topk:
        max_k = int(num_transfer_tokens[:, step_idx].max().item())
        if max_k <= 0:
            return transfer_index
        _, topk_idx = torch.topk(confidence, k=max_k, dim=-1, sorted=True)
        ranks = torch.arange(max_k, device=device).unsqueeze(0)
        valid = ranks < num_transfer_tokens[:, step_idx].unsqueeze(1)
        transfer_index.scatter_(1, topk_idx, valid)
        return transfer_index

    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens[j, step_idx].item())
        if k <= 0:
            continue
        _, select_index = torch.topk(confidence[j], k=k, sorted=False)
        transfer_index[j, select_index] = True
    return transfer_index


def _compute_logits_stats_chunked(
    model: AutoModel,
    input_ids: torch.Tensor,
    mask_index: torch.Tensor | None,
    mask_keep_index: torch.Tensor | None,
    temperature: float,
    chunk_size: int,
    approx_confidence: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算扩散采样所需的 argmax 与置信度，避免构造完整 logits。"""
    # 仅支持 cfg_scale=0 路径；input_ids: [B, L]
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
        return_dict=False,
        cache_position=None,
    )
    hidden_states = outputs[0]
    batch, seqlen, hidden = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, hidden)
    mask_flat = None
    if mask_index is not None:
        mask_flat = mask_index.reshape(-1)
        if mask_keep_index is not None:
            mask_flat = mask_flat & mask_keep_index.reshape(-1)
        if mask_flat.any():
            hidden_flat = hidden_flat[mask_flat]
        else:
            x0 = input_ids.clone()
            x0_p = torch.zeros(
                (batch, seqlen), dtype=torch.float32, device=hidden_states.device
            )
            return x0, x0_p

    lm_head = model.lm_head
    vocab = lm_head.weight.shape[0]
    chunk_size = min(chunk_size, vocab)

    device = hidden_states.device
    best_score = torch.full((hidden_flat.shape[0],), float("-inf"), device=device)
    best_idx = torch.zeros((hidden_flat.shape[0],), dtype=torch.long, device=device)
    best_logit = torch.zeros((hidden_flat.shape[0],), dtype=torch.float32, device=device)
    lse = None

    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        weight = lm_head.weight[start:end]
        bias = lm_head.bias[start:end] if lm_head.bias is not None else None
        logits_chunk = torch.nn.functional.linear(hidden_flat, weight, bias)
        logits_fp32 = logits_chunk.float()

        if temperature == 0:
            noisy = logits_fp32
        else:
            noise = torch.rand_like(logits_fp32)
            gumbel = (-torch.log(noise)).pow(temperature)
            noisy = logits_fp32.exp() / gumbel

        chunk_max, chunk_idx = noisy.max(dim=-1)
        update = chunk_max > best_score
        if update.any():
            best_score[update] = chunk_max[update]
            best_idx[update] = start + chunk_idx[update]
            best_logit[update] = logits_fp32[update, chunk_idx[update]]

        if not approx_confidence:
            chunk_lse = torch.logsumexp(logits_fp32, dim=-1)
            if lse is None:
                lse = chunk_lse
            else:
                lse = torch.logaddexp(lse, chunk_lse)

    if mask_flat is None:
        x0 = best_idx.view(batch, seqlen)
        if approx_confidence:
            x0_p = best_logit.view(batch, seqlen)
        else:
            x0_p = torch.exp((best_logit - lse).view(batch, seqlen))
    else:
        x0_flat = input_ids.view(-1).clone()
        x0_flat[mask_flat] = best_idx
        x0 = x0_flat.view(batch, seqlen)
        x0_p_flat = torch.zeros(
            (batch * seqlen,), dtype=torch.float32, device=device
        )
        if approx_confidence:
            x0_p_flat[mask_flat] = best_logit
        else:
            x0_p_flat[mask_flat] = torch.exp(best_logit - lse)
        x0_p = x0_p_flat.view(batch, seqlen)
    return x0, x0_p


def _compute_logits_stats_full(
    model: AutoModel,
    input_ids: torch.Tensor,
    mask_index: torch.Tensor | None,
    mask_keep_index: torch.Tensor | None,
    temperature: float,
    approx_confidence: bool,
    gumbel_fp32: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算扩散采样所需的 argmax 与置信度，构造完整 logits。"""
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        output_router_logits=False,
        return_dict=False,
        cache_position=None,
    )
    hidden_states = outputs[0]
    batch, seqlen, hidden = hidden_states.shape
    hidden_flat = hidden_states.reshape(-1, hidden)
    mask_flat = None
    if mask_index is not None:
        mask_flat = mask_index.reshape(-1)
        if mask_keep_index is not None:
            mask_flat = mask_flat & mask_keep_index.reshape(-1)
        if mask_flat.any():
            hidden_flat = hidden_flat[mask_flat]
        else:
            x0 = input_ids.clone()
            x0_p = torch.zeros(
                (batch, seqlen), dtype=torch.float32, device=hidden_states.device
            )
            return x0, x0_p

    lm_head = model.lm_head
    logits = torch.nn.functional.linear(hidden_flat, lm_head.weight, lm_head.bias)
    logits_fp32 = logits.float()
    logits_with_noise = add_gumbel_noise(
        logits_fp32, temperature=temperature, gumbel_fp32=gumbel_fp32
    )
    x0_masked = torch.argmax(logits_with_noise, dim=-1)
    if approx_confidence:
        x0_p_masked = torch.squeeze(
            torch.gather(
                logits_fp32, dim=-1, index=torch.unsqueeze(x0_masked, -1)
            ),
            -1,
        )
    else:
        p = torch.softmax(logits_fp32, dim=-1)
        x0_p_masked = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0_masked, -1)), -1
        )

    if mask_flat is None:
        x0 = x0_masked.view(batch, seqlen)
        x0_p = x0_p_masked.view(batch, seqlen)
    else:
        x0_flat = input_ids.view(-1).clone()
        x0_flat[mask_flat] = x0_masked
        x0 = x0_flat.view(batch, seqlen)
        x0_p_flat = torch.zeros(
            (batch * seqlen,), dtype=torch.float32, device=hidden_states.device
        )
        x0_p_flat[mask_flat] = x0_p_masked
        x0_p = x0_p_flat.view(batch, seqlen)
    return x0, x0_p


def _apply_threshold_mask(
    transfer_index: torch.Tensor,
    confidence: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if threshold <= 0.0:
        return transfer_index
    threshold_mask = confidence >= threshold
    if not threshold_mask.any():
        return transfer_index
    batch_has = threshold_mask.any(dim=1, keepdim=True)
    return torch.where(batch_has, transfer_index & threshold_mask, transfer_index)


@torch.no_grad()
def diffusion_generate(
    model: AutoModel,
    prompt_ids: torch.Tensor,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    mask_id: int = 156895,
    logits_chunk_size: int = 0,
    masked_lm_head: bool = False,
    vectorized_topk: bool = False,
    use_where_update: bool = False,
    masked_block_only: bool = False,
    approx_confidence: bool = False,
    gumbel_fp32: bool = False,
    block_topk_only: bool = False,
    threshold: float = 0.0,
    prefix_look: int = 0,
    after_look: int = 0,
) -> torch.Tensor:
    """
    直接移植自模型 README 的扩散采样逻辑。
    支持 batch>1，但需保证 prompt 等长且 padding 兼容。
    """
    assert prompt_ids.dim() == 2, "prompt_ids 应为 [B, L]"
    device = model.device
    batch_size = prompt_ids.size(0)
    x = torch.full(
        (batch_size, prompt_ids.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, : prompt_ids.shape[1]] = prompt_ids.clone()
    prompt_index = x != mask_id
    transfer_index_buf = torch.zeros_like(x, dtype=torch.bool, device=device)
    transfer_index_block_buf = torch.zeros(
        (batch_size, block_length), dtype=torch.bool, device=device
    )

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt_ids.shape[1]
                + num_block * block_length : prompt_ids.shape[1]
                + (num_block + 1) * block_length,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        block_start = prompt_ids.shape[1] + num_block * block_length
        block_end = prompt_ids.shape[1] + (num_block + 1) * block_length
        for i in range(steps_per_block):
            mask_index = x == mask_id
            mask_keep_index = None
            if masked_block_only:
                mask_keep_index = torch.zeros_like(mask_index)
                mask_keep_index[:, block_start:block_end] = True
            transfer_window = None
            if prefix_look > 0 or after_look > 0:
                window_start = max(prompt_ids.shape[1], block_start - prefix_look)
                window_end = min(prompt_ids.shape[1] + gen_length, block_end + after_look)
                if window_end > window_start:
                    transfer_window = torch.zeros_like(mask_index)
                    transfer_window[:, window_start:window_end] = True
            if logits_chunk_size > 0:
                if cfg_scale > 0.0:
                    raise NotImplementedError("chunked logits 暂不支持 cfg_scale > 0")
                x0, x0_p = _compute_logits_stats_chunked(
                    model,
                    x,
                    mask_index=mask_index if masked_lm_head else None,
                    mask_keep_index=mask_keep_index,
                    temperature=temperature,
                    chunk_size=logits_chunk_size,
                    approx_confidence=approx_confidence,
                )
            elif cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                logits_with_noise = add_gumbel_noise(
                    logits, temperature=temperature, gumbel_fp32=gumbel_fp32
                )
                x0 = torch.argmax(logits_with_noise, dim=-1)
                if approx_confidence:
                    logits_fp32 = logits.float()
                    x0_p = torch.squeeze(
                        torch.gather(
                            logits_fp32, dim=-1, index=torch.unsqueeze(x0, -1)
                        ),
                        -1,
                    )
                else:
                    p = torch.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
            else:
                if masked_lm_head:
                    x0, x0_p = _compute_logits_stats_full(
                        model,
                        x,
                        mask_index=mask_index,
                        mask_keep_index=mask_keep_index,
                        temperature=temperature,
                        approx_confidence=approx_confidence,
                        gumbel_fp32=gumbel_fp32,
                    )
                else:
                    logits = model(x).logits

                    logits_with_noise = add_gumbel_noise(
                        logits, temperature=temperature, gumbel_fp32=gumbel_fp32
                    )
                    x0 = torch.argmax(logits_with_noise, dim=-1)
                    if approx_confidence:
                        logits_fp32 = logits.float()
                        x0_p = torch.squeeze(
                            torch.gather(
                                logits_fp32, dim=-1, index=torch.unsqueeze(x0, -1)
                            ),
                            -1,
                        )
                    else:
                        p = torch.softmax(logits, dim=-1)
                        x0_p = torch.squeeze(
                            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)),
                            -1,
                )

            if steps_per_block == 0:
                continue
            if i >= num_transfer_tokens.shape[1]:
                continue

            if num_block < num_blocks - 1:
                x0[..., prompt_ids.shape[1] + (num_block + 1) * block_length :] = mask_id

            x0_p[:, prompt_ids.shape[1] + (num_block + 1) * block_length :] = float("-inf")

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, float("-inf"))
            if transfer_window is not None:
                confidence = torch.where(transfer_window, confidence, float("-inf"))

            if block_topk_only:
                confidence_block = confidence[:, block_start:block_end]
                transfer_index_block = _build_transfer_index(
                    confidence_block,
                    num_transfer_tokens,
                    i,
                    vectorized_topk,
                    transfer_index_block_buf,
                )
                if threshold > 0.0 and not approx_confidence:
                    transfer_index_block = _apply_threshold_mask(
                        transfer_index_block, confidence_block, threshold
                    )
                if use_where_update:
                    x[:, block_start:block_end] = torch.where(
                        transfer_index_block,
                        x0[:, block_start:block_end],
                        x[:, block_start:block_end],
                    )
                else:
                    block_x = x[:, block_start:block_end]
                    block_x[transfer_index_block] = x0[:, block_start:block_end][
                        transfer_index_block
                    ]
                    x[:, block_start:block_end] = block_x
            else:
                transfer_index = _build_transfer_index(
                    confidence,
                    num_transfer_tokens,
                    i,
                    vectorized_topk,
                    transfer_index_buf,
                )
                if threshold > 0.0 and not approx_confidence:
                    transfer_index = _apply_threshold_mask(
                        transfer_index, confidence, threshold
                    )
                if use_where_update:
                    x = torch.where(transfer_index, x0, x)
                else:
                    x[transfer_index] = x0[transfer_index]

    return x


def run_eval(args) -> Tuple[float, Dict[str, float]]:
    device = resolve_device(args.device)
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer
    seed_execution(args.seed, device)
    dtype = torch.bfloat16 if device.type == "npu" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, revision=args.revision)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, revision=args.revision, torch_dtype=dtype, device_map=None
    ).to(device)
    model.eval()

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

        for i in range(0, total, args.batch_size):
            batch_prompts = prompts[i : i + args.batch_size]
            batch_labels = labels[i : i + args.batch_size]
            for prompt_text, label in zip(batch_prompts, batch_labels):
                prompt_ids = tokenizer(prompt_text)["input_ids"]
                prompt_ids = torch.tensor(prompt_ids, device=device).unsqueeze(0)
                out_ids = diffusion_generate(
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
                gen_text = tokenizer.batch_decode(
                    out_ids[:, prompt_ids.shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                pred = extract_choice(gen_text)
                if pred == label:
                    correct += 1

        acc = correct / total if total > 0 else 0.0
        subject_acc[subject] = acc
        print(f"[{subject}] acc={acc:.4f} ({correct}/{total})")
        sys.stdout.flush()

    macro_acc = sum(subject_acc.values()) / len(subject_acc)
    return macro_acc, subject_acc


def main():
    parser = argparse.ArgumentParser(description="CEval 扩散解码精度评测")
    parser.add_argument("--model", required=True, help="HF 模型路径")
    parser.add_argument("--revision", default=None, help="Model revision (pin a commit for remote weights)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="npu:0", help="设备，默认 npu:0；不可用则报错，cpu 必须显式指定")
    parser.add_argument("--dataset_dir", default=None, help="Optional CEval cache directory/HF_HOME")
    parser.add_argument("--split", default="val", choices=["dev", "val", "test"], help="评测 split")
    parser.add_argument("--batch_size", type=int, default=1, help="解码 batch，暂仅建议 1")
    parser.add_argument("--steps", type=int, default=24, help="扩散步数")
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
    parser.add_argument("--log_file", default="", help="日志输出路径（可选）")
    args = parser.parse_args()

    if args.dataset_dir:
        os.environ.setdefault("HF_HOME", args.dataset_dir)

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        log_f = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = log_f
        sys.stderr = log_f

    macro_acc, subject_acc = run_eval(args)
    print("\n=== CEval 扩散解码评测结果 ===")
    print(f"宏平均准确率: {macro_acc:.4f}")
    for k, v in sorted(subject_acc.items()):
        print(f"{k}\t{v:.4f}")
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
