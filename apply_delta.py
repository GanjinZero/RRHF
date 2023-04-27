"""
From fastchat
Apply the delta weights on top of a base model.
Usage:
python apply_delta.py --base ~/model_weights/llama-7b --target ~/model_weights/wombat-7b --delta GanjinZero/wombat-7b-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def apply_delta(base_model_path, target_model_path, delta_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print(f"Loading the delta from {delta_path}")
    delta = AutoModelForCausalLM.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    delta_tokenizer = AutoTokenizer.from_pretrained(delta_path, use_fast=False)

    DEFAULT_PAD_TOKEN = "[PAD]"
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    num_new_tokens = base_tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))

    base.resize_token_embeddings(len(base_tokenizer))
    input_embeddings = base.get_input_embeddings().weight.data
    output_embeddings = base.get_output_embeddings().weight.data
    input_embeddings[-num_new_tokens:] = 0
    output_embeddings[-num_new_tokens:] = 0

    print("Applying the delta")
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    print(f"Saving the target model to {target_model_path}")
    base.save_pretrained(target_model_path)
    delta_tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)

    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
