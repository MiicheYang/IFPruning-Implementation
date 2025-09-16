import torch
import torch.nn as nn
from transformers import AutoConfig

from src.llm_with_mask import LlamaForCausalLMWithMask
from src.sparsity_prediction_model import SparsityPredictor


class IFPruningLLM(nn.Module):
    def __init__(self,
        hf_model_name: str = "meta-llama/Llama-3.1-8B",
        padding_idx: int = 128004,
    ):
        super().__init__()
        self.llm_config = AutoConfig.from_pretrained(hf_model_name)

        self.llm = LlamaForCausalLMWithMask.from_pretrained(hf_model_name)
        self.sparsity_predictor = SparsityPredictor(
            hidden_dim=self.llm_config.hidden_size,
            num_layers=self.llm_config.num_hidden_layers,
            ffn_dim=self.llm_config.intermediate_size,
            padding_idx=padding_idx,
        )
    def forward(self, input_ids):
        # batch_size, source_llm_num_layers, source_llm_ffn_dim
        ffn_mask: torch.Tensor = self.sparsity_predictor(input_ids)
        # source_llm_num_layers, batch_size, 1, source_llm_ffn_dim
        ffn_mask = ffn_mask.transpose(0, 1).unsqueeze(2)
        outputs = self.llm(
            input_ids=input_ids,
            ffn_mask_all_layers=ffn_mask,
        )
        return outputs