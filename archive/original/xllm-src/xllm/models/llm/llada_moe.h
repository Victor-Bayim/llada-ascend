/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "qwen3_moe.h"

namespace xllm {

// Reuse Qwen3 MoE implementation for LLaDA-MoE series (config-compatible).
// Key differences are driven by config: hidden_size/head count, vocab size,
// FFN sizes, rope theta/max position, etc. Loaded via dedicated args loader.
using LLaDAMoEForCausalLM = Qwen3MoeForCausalLM;

// Register causal model under "llada"
REGISTER_CAUSAL_MODEL(llada, Qwen3MoeForCausalLM);

// Register model args loader for LLaDA config schema.
REGISTER_MODEL_ARGS(llada, [&] {
  LOAD_ARG_OR(model_type, "model_type", "llada");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(attention_bias, "attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "attention_dropout", 0.0f);
  LOAD_ARG_OR(bos_token_id, "bos_token_id", 156892);  // not provided, fall back
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 156892);
  LOAD_ARG_OR(pad_token_id, "pad_token_id", 156892);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "hidden_size", 2048);
  LOAD_ARG_OR(use_qk_norm, "qk_layernorm", false);
  // derive head_dim if not explicitly provided
  LOAD_ARG_OR(n_heads, "num_attention_heads", 16);
  LOAD_ARG_OR(n_kv_heads, "num_key_value_heads", args->n_heads());
  LOAD_ARG_OR(head_dim, "head_dim", args->hidden_size() / args->n_heads());
  LOAD_ARG_OR(initializer_range, "initializer_range", 0.02f);
  // dense FFN and MoE expert sizes in LLaDA config
  LOAD_ARG_OR(intermediate_size, "dense_intermediate_size", 8192);
  LOAD_ARG_OR(moe_intermediate_size, "expert_intermediate_size", 1024);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 8192);
  LOAD_ARG_OR(rope_theta, "rope_theta", 50000.0f);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-5);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 16);
  LOAD_ARG_OR(num_experts, "num_experts", 64);
  // ATB MoE 内核在大 top-k 下不稳定，先降为 top-1 以测试可行路径。
  LOAD_ARG_OR(num_experts_per_tok, "num_experts_per_tok", 1);
  LOAD_ARG_OR(output_router_logits, "output_router_logits", false);
  LOAD_ARG_OR(router_aux_loss_coef, "router_aux_loss_coef", 0.01f);
  LOAD_ARG_OR(vocab_size, "vocab_size", 157184);
  // keep defaults from qwen3_moe for remaining fields
  LOAD_ARG_OR(decoder_sparse_step, "decoder_sparse_step", 1);
  LOAD_ARG_OR(norm_topk_prob, "norm_topk_prob", true);
  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(sliding_window, "sliding_window", args->max_position_embeddings());
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR(attention_bias, "attention_bias", args->attention_bias());

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

// Force optimizer to keep the static registration symbols.
static const bool llada_force_register =
    llada_registered && llada_args_loader_registered;

}  // namespace xllm
