/* Fallback MoE implementation using torch operations (no ATB). */

#include "fused_moe.h"

#include <algorithm>

namespace xllm {
namespace layer {
namespace fallback {

FbFusedMoEImpl::FbFusedMoEImpl(int num_experts,
                               int top_k,
                               int hidden_size,
                               int intermediate_size,
                               bool is_gated,
                               bool has_score_bias,
                               bool has_bias,
                               const std::string& hidden_act,
                               const QuantArgs& quant_args,
                               const ParallelArgs& parallel_args,
                               const torch::TensorOptions& options)
    : num_experts_(num_experts),
      topk_(top_k),
      hidden_size_(hidden_size),
      intermediate_size_(intermediate_size),
      is_gated_(is_gated),
      has_score_bias_(has_score_bias),
      has_bias_(has_bias),
      hidden_act_(hidden_act),
      quant_args_(quant_args),
      options_(options) {
  gate_ = register_module(
      "gate",
      FbReplicatedLinear(hidden_size_, num_experts_, has_score_bias_,
                         quant_args_, options_));
  experts_.reserve(num_experts_);
  for (int i = 0; i < num_experts_; ++i) {
    auto mlp = FbDenseMLP(hidden_size_, intermediate_size_, is_gated_,
                          has_bias_, hidden_act_, quant_args_, parallel_args,
                          options_);
    experts_.push_back(register_module("expert_" + std::to_string(i), mlp));
  }
}

torch::Tensor FbFusedMoEImpl::forward(const torch::Tensor& hidden_states,
                                      const ModelInputParams&) {
  // hidden_states: [tokens, hidden_size]
  auto router_logits = gate_->forward(hidden_states);  // [tokens, num_experts]
  auto topk = std::min(topk_, num_experts_);
  auto topk_result = torch::topk(router_logits, topk, /*dim=*/-1);
  auto topk_weights = std::get<0>(topk_result);
  auto topk_indices = std::get<1>(topk_result);  // [tokens, topk]
  // softmax over selected experts
  topk_weights = torch::softmax(topk_weights, /*dim=*/-1);

  auto output = torch::zeros(
      {hidden_states.size(0), hidden_size_},
      torch::TensorOptions().dtype(hidden_states.dtype()).device(
          hidden_states.device()));

  // Scatter compute per expert
  for (int expert = 0; expert < num_experts_; ++expert) {
    auto mask = (topk_indices == expert);  // [tokens, topk] bool
    if (!mask.any().item<bool>()) {
      continue;
    }
    // collect tokens routed to this expert
    auto selected = torch::nonzero(mask.sum(-1)).squeeze(-1);
    if (selected.numel() == 0) continue;
    auto expert_in = hidden_states.index_select(0, selected);
    auto expert_out = experts_[expert]->forward(expert_in);
    // gather weights for selected tokens
    auto sel_mask = mask.index_select(0, selected);
    auto sel_weights = (topk_weights.index_select(0, selected) * sel_mask)
                           .sum(-1, true);  // [selected,1]
    expert_out = expert_out * sel_weights;
    output.index_add_(0, selected, expert_out);
  }
  return output;
}

void FbFusedMoEImpl::load_state_dict(const StateDict& state_dict) {
  gate_->load_state_dict(state_dict.get_dict_with_prefix("gate."));
  for (int i = 0; i < num_experts_; ++i) {
    experts_[i]->load_state_dict(
        state_dict.get_dict_with_prefix("experts." + std::to_string(i) + "."));
  }
}

}  // namespace fallback
}  // namespace layer
}  // namespace xllm
