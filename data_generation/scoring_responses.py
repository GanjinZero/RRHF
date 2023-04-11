#### The code is modified from trlX
import json
import math
import os
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def create_reward_fn(): 
    reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"

    class RewardModel(nn.Module):
        def __init__(self, checkpoint_path, eos_token_id):
            super().__init__()
            model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
            self.transformer = model.transformer
            self.v_head = nn.Linear(model.config.n_embd, 1, bias=False)
            self.eos_token_id = eos_token_id

        def forward(self, input_ids):
            states = self.transformer(input_ids)[0]
            rewards = self.v_head(states).squeeze(-1)
            ends = torch.argmax((input_ids == self.eos_token_id).float(), dim=1).view(-1, 1)
            returns = torch.gather(rewards, 1, ends).squeeze(-1)
            return returns

    reward_model = RewardModel("EleutherAI/gpt-j-6B", reward_tokenizer.eos_token_id)
    directory = "Dahoas/gptj-rm-static" # 
    for fpath in os.listdir(directory):
        if fpath.endswith(".pt") or fpath.endswith(".bin"):
            checkpoint = os.path.join(directory, fpath)
            break
    ckpt_state = torch.load(checkpoint)
    ckpt_state = {k:v for k, v in ckpt_state.items() if not k.startswith('model.')}
    reward_model.load_state_dict(ckpt_state)
    reward_model.eval()
    reward_model.requires_grad_(False)
    device = 'cuda:0'
    reward_model = reward_model.half().to(device)

    def reward_fn(samples):
        samples = [s + reward_tokenizer.eos_token for s in samples]
        input = reward_tokenizer(samples, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(
            device
        )

        mbs = 24
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            input_ids = input.input_ids[batch_ixs]
            rewards = reward_model(input_ids)
            out.extend(rewards.cpu().tolist())

        return out

    return reward_fn

import sys

with open(sys.argv[1], 'r') as f:
    candidates = [json.loads(item.strip()) for item in f.readlines()]
outputs = []
input_buffer = []
response_num = len(candidates[0][1]) + 2
reward_fn = create_reward_fn()

for idx in tqdm(range(len(candidates))):
    input_buffer.append([candidates[idx][0] + ' ' + item for item in candidates[idx][1]])
    input_buffer[-1].append(candidates[idx][0] + ' ' + candidates[idx][2])
    input_buffer[-1].append(candidates[idx][0] + ' ' + candidates[idx][3])
    if len(input_buffer) == 5 or idx == len(candidates)-1:
        input_texts = sum(input_buffer, [])
        reward_results = reward_fn(input_texts)
        for i in range(0, len(reward_results), response_num):
            rs = reward_results[i: i+response_num]
            outputs.append(rs)
        input_buffer = []

assert len(outputs) == len(candidates)

finals = []
for rs, cans in tqdm(zip(outputs, candidates)):
    finals.append({'prompt':cans[0], 'response':cans[1]+[cans[2], cans[3]], 'scores':rs})
    assert len(finals[-1]['response']) == len(finals[-1]['scores'])

with open(sys.argv[2], 'w') as f:
    json.dump(finals, f, indent=2)

