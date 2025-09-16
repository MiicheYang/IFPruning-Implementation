"""
python baseline_inference.py --eval_model_name=source_llm \
    --input_length=2000 \
    --output_length=1000
"""
import csv
import os
import time
from typing import Dict, List

import numpy as np
import torch
import datasets
from absl import app, flags
from tqdm.auto import tqdm as auto_tqdm
from transformers import (
    AutoModelForCausalLM,
    GenerationConfig,
    LlamaForCausalLM,
)

from src.ifpruning_config import (
    IFPRUNING_CONFIG_V1,
    SOURCE_LLM_CONFIG,
    INTERNAL_LM_3B_CONFIG,
    INTERNAL_LM_9B_CONFIG,
)

flags.DEFINE_enum(
    "eval_model_name",
    None,
    enum_values=["source_llm", "small_llm", "moe", "internal_9b", "internal_3b"],
    help="Which model is being evaluated, "
    "dense model, MoE, or the internal model used in the paper",
)
flags.DEFINE_integer(
    "input_length",
    None,
    help="The input length for LLM generation. All inputs will be padded to this value.",
    required=True,
)
flags.DEFINE_integer(
    "output_length",
    None,
    help=(
        "The output length for LLM generation."
        " The LLM is required to generation until this length.",
    ),
    required=True,
)
flags.DEFINE_boolean(
    "do_generate",
    False,
    help="Whether perform text generation."
)

FLAGS = flags.FLAGS

def main(argv):
    csv_file = f"logs/{FLAGS.eval_model_name}.csv"
    print(f"Save the logs to {csv_file}")
    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))

    device = torch.device("cuda")
    start_model_load = time.perf_counter()
    if FLAGS.eval_model_name == "source_llm":
        llm_config = SOURCE_LLM_CONFIG
        inference_llm = LlamaForCausalLM(
            llm_config,
        ).bfloat16().to(device)
    elif FLAGS.eval_model_name == "small_llm":
        llm_config = IFPRUNING_CONFIG_V1
        inference_llm = LlamaForCausalLM(
            llm_config,
        ).bfloat16().to(device)
    elif FLAGS.eval_model_name == "internal_3b":
        llm_config = INTERNAL_LM_3B_CONFIG
        inference_llm = LlamaForCausalLM(
            llm_config,
        ).bfloat16().to(device)
    elif FLAGS.eval_model_name == "internal_9b":
        llm_config = INTERNAL_LM_9B_CONFIG
        inference_llm = LlamaForCausalLM(
            llm_config,
        ).bfloat16().to(device)

    elif FLAGS.eval_model_name == "moe":
        model_name = "Qwen/Qwen1.5-MoE-A2.7B"
        inference_llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        ).to(device)
    else:
        raise ValueError()
    end_model_load = time.perf_counter()
    print(f"Time for loading the model: {end_model_load-start_model_load}")
    vocab_size = inference_llm.config.vocab_size

    inference_llm.requires_grad_(False)


    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=FLAGS.output_length,
        min_new_tokens=FLAGS.output_length,
        eos_token_id=None,
        num_return_sequences=4,
    )

    total_params = sum(p.numel() for p in inference_llm.parameters())
    print(f"Total parameters: {total_params:,}")

    dataset = datasets.load_dataset("google/IFEval")["train"]
    input_texts: List[str] = dataset["prompt"][:110]

    ttft_list = []
    generation_time_list = []

    prog_bar = auto_tqdm(range(len(input_texts)))

    for i, text in enumerate(input_texts):
        input_ids = torch.randint(
            low=0, high=vocab_size, size=(1, FLAGS.input_length)
        ).long().to(device)
        # input_ids = torch.LongTensor(input_ids).view(1, -1).to(device)

        # Step 1: Prefill only (outside generate)
        torch.cuda.synchronize()
        start_prefill = time.perf_counter()
        _ = inference_llm(input_ids=input_ids)
        torch.cuda.synchronize()
        end_prefill = time.perf_counter()
        prefill_time = end_prefill - start_prefill
        ttft_list.append(prefill_time)

        if FLAGS.do_generate:
            # Step 2: Full generate() (includes prefill + generation)
            torch.cuda.synchronize()
            start_gen = time.perf_counter()
            _ = inference_llm.generate(
                input_ids=input_ids,
                generation_config=generation_config,
            )
            torch.cuda.synchronize()
            end_gen = time.perf_counter()
            total_gen_time = end_gen - start_gen

            # Step 3: Isolate generation time
            actual_generation_time = total_gen_time - prefill_time
            generation_time_list.append(actual_generation_time)
        prog_bar.update(1)

    avg_ttft = np.mean(ttft_list[50:])
    if FLAGS.do_generate:
        avg_gen_time = np.mean(generation_time_list[50:])
        tps = FLAGS.output_length / avg_gen_time
    else:
        avg_gen_time = 0.0
        tps = 0.0

    print("Averaget TTFT: ", avg_ttft)
    print("Averaget generation time: ", avg_gen_time)
    print("Averaget TPS: ", tps)

    # Prepare row data
    row = {
        "input_length": FLAGS.input_length,
        "output_length": FLAGS.output_length,
        "ttft": avg_ttft,
        "generation_time": avg_gen_time,
        "tps": tps,
    }

    # Check if the file exists to decide whether to write header
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"📁 Logged results to: {csv_file}")


if __name__ == "__main__":
    app.run(main)