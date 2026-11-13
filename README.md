# LLaDA-MoE-7B on Ascend (Huawei ICT)

Adaptation of the LLaDA diffusion language model to Ascend 910B.

## Overview

LLaDA generates by iteratively filling masked tokens. This work adapts the model to Ascend via torch_npu, with changes focused on vocabulary projection chunking and masked-position lm_head to reduce memory pressure on long sequences and larger batches.

## Role

Team captain. Independent implementation of the diffusion inference adaptation, memory-path changes, and evaluation for this problem.

## Key Points

- Vocabulary projection chunking
- Masked-position lm_head
- Evaluation across 14 input shapes (32–2048 tokens, batch 1 and 16) on single NPU

## Structure

- `analysis/llada_torch_npu/` — main adaptation code
- `xllm-src/` — related framework modifications
- `analysis/final_report/` — summary notes

## Notes

Datasets used for evaluation (e.g. CEval) are not included; refer to the official sources. Results are from the competition evaluation environment.

## License

MIT (or specify). Attribution to upstream frameworks retained where applicable.
