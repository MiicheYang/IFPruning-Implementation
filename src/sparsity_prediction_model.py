import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM


class SparsityPredictor(nn.Module):
    def __init__(
        self,
        hf_model_name: str,
        num_layers: int,
        ffn_dim: int,
        padding_idx: int,
        pruned_ffn_dim: int,
    ):
        """
        The sparsity predictor for IFPruning.
        Args:
            hidden_dim: the hidden dimension of the source LLM to be pruned.
            num_layers: the number of transformer layers of the source LLM.
            ffn_dim: the FFN intermediate dimension of the source LLM.
            padding_idx: the index of the padding token.
            pruned_ffn_dim: the FFN intermediate dimension after pruning.
        """
        super().__init__()
        self.source_llm_num_layers = num_layers
        self.source_llm_ffn_dim = ffn_dim

        self.feature_extractor = Qwen2ForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_2",
        )
        hidden_dim = self.feature_extractor.config.hidden_size
        self.prediction_head1 = nn.Linear(hidden_dim, 128)
        self.prediction_head2 = nn.Linear(128, num_layers * ffn_dim)
        self.padding_idx = padding_idx
        self.topk = pruned_ffn_dim

    def topk_operator(self, importance_scores: torch.Tensor):
        """
        Args:
            importance_scores: a tensor with shape
                [batch_size, source_llm_num_layers, source_llm_ffn_dim]
        Returns:
            selected_indices: a tnesor with shape
                [batch_size, source_llm_num_layers, top_k_indices]]
        """
        bsz, source_llm_num_layers, _ = importance_scores.size()
        importance_scores = importance_scores.view(-1, importance_scores.size(-1))
        # topk_indices = torch.topk(importance_scores, dim=1, k=topk, sorted=False).indices
        # topk_indices = topk_indices.view(bsz, source_llm_num_layers, topk)
        # return topk_indices

        topk_vals = torch.topk(importance_scores, k=self.topk, dim=1).values
        thresholds = topk_vals[:, -1].unsqueeze(1)

        # construct mask
        mask = (importance_scores >= thresholds)
        mask = mask.view(bsz, source_llm_num_layers, -1)
        return mask

    def forward(self, input_ids):
        outputs = self.feature_extractor(input_ids=input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        non_pad_mask = input_ids != self.padding_idx
        last_non_pad_indices = non_pad_mask.sum(dim=1) - 1

        batch_size = input_ids.size(0)
        last_hidden_states = hidden_states[torch.arange(batch_size), last_non_pad_indices, :]

        intermediate = self.prediction_head1(last_hidden_states)
        predictions = self.prediction_head2(intermediate)

        # batch_size, source_llm_num_layers, source_llm_ffn_dim
        ffn_importance_scores = predictions.view(batch_size, -1, self.source_llm_ffn_dim)

        ffn_selection_mask = self.topk_operator(ffn_importance_scores)
        return ffn_selection_mask