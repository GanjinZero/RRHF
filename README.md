
# Wombat 🐻‍❄️: from RLHF to RRHF, Aligning Human Preferences in a 'Right' Way

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

**License Notices**:  The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

## Overview

This is the repository for RRHF (**R**ank **R**esponse from **H**uman **F**eedback). RRHF helps align large language models with human perference easier. 

Reinforcement learning from human feedback (RLHF) enables the alignment of large language models with human preferences, which can extremely improve the quality of interactions between humans and language models.
Recent practice of RLHF uses PPO to enable the large language model optimization of such alignment. However, implementing PPO is non-trivial (where the training procedure requires interactive between policy, behavior policy, reward, value model) and it is also tedious to tuning many hyper-parameters.
Our motivation is to simplifiy the alignment between language models with human preference, and our proposed paradigm RRHF (**R**ank **R**esponse from **H**uman **F**eedback) can achieve such alignment as easily as conventional fine-tuning.
It is simpler than PPO from the aspects of coding, model counts, and hyperparameters.

In our preliminary experiments, we compare RRHF and PPO using 7B LLaMA [1] and Alpaca [2] models on Anthropic’s Helpful and Harmless (HH) [3] dataset. We evaluate the results by perplexity (PPL) and reward model scores (Reward). 
With a much simplier training paradigm, we found that RRHF can achieve slightly better result than PPO in terms of generation fluency (PPL) and alignements (Reward).

| Models| Setting  | PPL       | Reward    |
|--------|---------|-----------|-----------|
| LLaMA  | PPO     | 42.53     | -1.62     |
| Alpaca | PPO     | **13.84** | *-1.03*   |
| LLaMA  | RRHF    | 67.12     | -1.34     |
| Alpaca | RRHF    | *14.75*   | **-1.02** |


For details, please refer to our [paper](RRHF.pdf). RRHF is still working in progress, and there are still limitations in this preliminary study.
Due to the large cost of human evaluation, we experiment on the HH datasets and use a trained reward model *Dahoas/gptj-rm-static* trained by [Dahoas](https://github.com/Dahoas/reward-modeling.git). 
The reward model plays a role of a synthetic human feedback and the experiments is a proof-of-concept for RRHF.
We are open to any suggestions and discussions and feel free to contact us through yuanzheng.yuanzhen@alibaba-inc.com or yuanhy20@mails.tsinghua.edu.cn.

[1]: LLaMA: Open and Efficient Foundation Language Models. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. https://arxiv.org/abs/2302.13971v1

[2]: Stanford alpaca: An instruction-following llama model. Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. https://github.com/tatsu-lab/stanford_alpaca

[3]: HH: Training a helpful and harmless assistant with reinforcement learning from human feedback. Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. https://arxiv.org/abs/2204.05862

## Setting Up

To set up, you can use the following command lines to set up python3.8 and pytorch requirements:
```bash
conda create -n rrhf python=3.8
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
Then install Huggingface's transformers from the github repo for LLaMA models. 
```bash
git clone https://github.com/huggingface/transformers.git
pip install -e ./transformers
``` 
Install other packages:
```bash
pip install -r requirements.txt
``` 

## Data Generation

RRHF firstly samples responses for each query in the training data from the initial models, and then scores each response (including the 'chosen' and 'rejected' response in orginal HH labels) using the reward models.

The scripts for data generation are in [./data_generation](./data_generation), you can use throught the command line:
```bash
cd ./data_generation/
bash response_gen.sh <path_to_alpaca/hf_llama_directory> <path_to_data_directory>
```

We also release our generated data for the ease of RRHF training implementation through [this link](https://drive.google.com/file/d/1nAfBt0ldSy7m5O-Sgt05SQ1rK__NmC2Z/view?usp=sharing). After download, place it to <path_to_data_directory>.

## Training with RRHF

You can train your own model with generated or released datasets using the script [train.sh](./train.sh), please note that the training process requires 8*A100 80GB GPUs, bf16 and FSDP.
In the future, we will try efficient training methods such as LoRA or Prefix-tuning or Adapter to lower the computational resource requirements.

```bash
bash ./train.sh <path_to_alpaca_directory> <save_path_directory> <path_to_data_json>
```

## Wombat: build your own chatbot.

### Introduction

To produce a more general purpose language model chatbot, we introduce **Wombat** to the model zoo of open-resourced language models. 
We reuse the query from Alpaca training data, and sample responses from Alpaca, LLaMA, chatGPT and text-davinci-003.
We acquire quality assessments of responses from chatGPT, and train Alpace with RRHF to become a 'Wombat'.

<center>
    <a href="https://en.wikipedia.org/wiki/Wombat" target="_blank"><img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./wombat.png"></a>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Wombats are adorable little creatures native to Australia. The first three pictures are generated from Stable Diffusion.</div>
</center>

**An generated introduction of wombats by our 'Wombat':**

**Query:** Give a introduction of wombats

**Wombat's response:** Wombat is a marsupial found in Australia, Tasmania, and New Zealand. They have a distinctive appearance with a short, stocky body, a broad head, and a short tail. Wombats are herbivores and primarily graze on grasses, leaves, and bark. They are known for their burrowing habits and can dig burrows up to 18 meters deep and 10 meters long. Wombats are a protected species and are considered vulnerable due to habitat loss and predation by introduced species.

### Data and Training

You can acquire the data with rewards from [this link](https://drive.google.com/file/d/1_A0Fewde9Glbm4pzdYsGI8w0df5tHEG7/view?usp=sharing), and start training your own "**Wombat**". 
Use the following command lines:

```bash
bash ./train.sh <path_to_alpaca_directory> <save_path_directory> <path_to_data_json>
```

## Citation

For more informations, you can refer to our [paper](./paper/rrhf.pdf), the arxiv version is on the way, stay tuned.

## Acknowledgements

Our implementation and experiments are based on the codes from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [trlX](https://github.com/CarperAI/trlx), we appreciate their open-resourced codes and [LLaMA](https://arxiv.org/abs/2302.13971v1) to promote democratized NLP research, expecially for large lanague models.
We thank Tianhang Zhu to help collecting the data and constructive discussions.
