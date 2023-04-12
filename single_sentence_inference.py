import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from tqdm import trange
import json
import os
from stanford_alpaca.train import smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN

path = **path_to_your_model**
device = "cuda:7"

model = transformers.AutoModelForCausalLM.from_pretrained(path)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    '/nas-alinlp/zheng/llama/llama_hf/llama-7b',
    padding_side="left", # for batch decode
    use_fast=False,
)
if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        tokenizer=tokenizer,
        model=model,
    )
tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    }
)

model = model.to(device)

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"""

def generate_with_prompt_batch(instructs, inputs=None, batch_size=32, use_prompt=True, output_path=None):
    if inputs is None:
        inputs = [None] * len(instructs)

    results = []

    if output_path and os.path.exists(output_path):
        with open(output_path, 'r') as f:
            lines = f.readlines()
        lines = [line for line in lines if line]
        cnt = len(lines)
        print(f'Skip first {cnt} lines.')
        instructs = instructs[cnt:]
        inputs = inputs[cnt:]

    for batch_start in range(0, len(instructs), batch_size):
        batch_end = batch_start + batch_size
        batch_instructs = instructs[batch_start:batch_end]
        batch_inputs = inputs[batch_start:batch_end]

        batch_prompts = [
            generate_prompt(instr, inp) if use_prompt else instr
            for instr, inp in zip(batch_instructs, batch_inputs)
        ]

        encoded_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        input_ids = encoded_inputs["input_ids"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)

        if input_ids.shape[1] > 100:
            input_ids = input_ids[:,-100:]
            attention_mask = attention_mask[:,-100:]

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=500,
                temperature=1,
                num_beams=1,
                return_dict_in_generate=True,
                output_scores=True
            )

        for seq in generation_output.sequences:
            output = tokenizer.decode(seq)
            if use_prompt:
                try:
                    res = output.split("### Response:")[1].strip()
                except BaseException:
                    res = ''
            else:
                res = output
            results.append(res)
            if output_path:
                with open(output_path, 'a+') as f:
                    f.write(json.dumps({'response':res.split('</s>')[0], 'source':path}).strip() + "\n")

    results = [response.split('</s>')[0] for response in results]

    return results

def generate_with_prompt(instruct="What are the three primary colors?", input=None, use_prompt=True):
    results = generate_with_prompt_batch([instruct], [input], batch_size=1, use_prompt=use_prompt)
    return results[0]

while True:
    inp = input('Please input a query:')
    print(generate_with_prompt(inp, use_prompt=True))
