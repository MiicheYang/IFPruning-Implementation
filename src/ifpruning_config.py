from transformers.models.llama.configuration_llama import LlamaConfig

# Prune 8B to 3B
IFPRUNING_CONFIG_V1 = LlamaConfig(
    attn_implementation="flash_attention_2",
    vocab_size=128256,
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=128000,
    eos_token_id=128001,
    hidden_act="silu",
    hidden_size=4096,
    initializer_range=0.02,
    # prune `intermediate_size` from 14336 to 1536
    intermediate_size=1536,
    max_position_embeddings=131072,
    mlp_bias=False,
    model_type="llama",
    num_attention_heads=32,
    num_hidden_layers=32,
    num_key_value_heads=8,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling={
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    rope_theta=500000.0,
    tie_word_embeddings=False,
    torch_dtype="bfloat16",
    use_cache=True,
)

# lLAMA 8B
SOURCE_LLM_CONFIG = LlamaConfig(
    attn_implementation="flash_attention_2",
    vocab_size=128256,
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=128000,
    eos_token_id=128001,
    hidden_act="silu",
    hidden_size=4096,
    initializer_range=0.02,
    intermediate_size=14336,
    max_position_embeddings=131072,
    mlp_bias=False,
    model_type="llama",
    num_attention_heads=32,
    num_hidden_layers=32,
    num_key_value_heads=8,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling={
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    rope_theta=500000.0,
    tie_word_embeddings=False,
    torch_dtype="bfloat16",
    use_cache=True,
)

# The LLM reported in the paper, which is said to share the same architecture to LLAMA
# I use the arch configs in the appendix to config the model
INTERNAL_LM_9B_CONFIG = LlamaConfig(
    attn_implementation="flash_attention_2",
    vocab_size=100_000,
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=2048,
    initializer_range=0.02,
    intermediate_size=24576,
    max_position_embeddings=131072,
    mlp_bias=False,
    model_type="llama",
    num_attention_heads=16,
    num_hidden_layers=56,
    num_key_value_heads=2,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling={
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    rope_theta=500000.0,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
    use_cache=True,
)

INTERNAL_LM_3B_CONFIG = LlamaConfig(
    attn_implementation="flash_attention_2",
    vocab_size=100_000,
    attention_bias=False,
    attention_dropout=0.0,
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=2048,
    initializer_range=0.02,
    intermediate_size=6656,
    max_position_embeddings=131072,
    mlp_bias=False,
    model_type="llama",
    num_attention_heads=16,
    num_hidden_layers=56,
    num_key_value_heads=2,
    pretraining_tp=1,
    rms_norm_eps=1e-05,
    rope_scaling={
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    },
    rope_theta=500000.0,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
    use_cache=True,
)