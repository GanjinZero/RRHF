
# Wombat 🐻‍❄️: from RLHF to RRHF, Aligning Human Preferences in a 'Right' Way

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

<center>
    <a href="https://en.wikipedia.org/wiki/Wombat" target="_blank"><img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./assets/wombat.png"></a>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Wombats are adorable little creatures native to Australia. The first three pictures are generated from Stable Diffusion.</div>
</center>

**License Notices**:  The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

## Update:
-  2023/4/13 We have released the weights of Wombat - LLaMA on Hugging Face. One can recover Wombat weights based on it.

## Overview

This is the repository for RRHF (**R**ank **R**esponse to align **H**uman **F**eedback) and open-sourced language models Wombat. RRHF helps align large language models with human perference easier. 

Reinforcement learning from human feedback (RLHF) enables the alignment of large language models with human preferences, which can extremely improve the quality of interactions between humans and language models.
Recent practice of RLHF uses PPO to enable the large language model optimization of such alignment. However, implementing PPO is non-trivial (where the training procedure requires interactive between policy, behavior policy, reward, value model) and it is also tedious to tuning many hyper-parameters.
Our motivation is to simplifiy the alignment between language models with human preference, and our proposed paradigm RRHF (**R**ank **R**esponse from **H**uman **F**eedback) can achieve such alignment as easily as conventional fine-tuning.
It is simpler than PPO from the aspects of coding, model counts, and hyperparameters.


<center>
    <a target="_blank"><img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="./assets/comparison_of_workflow.png"></a>
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Overview of workflow comparison between PPO and RRHF.</div>
</center>

In our preliminary experiments, we compare RRHF and PPO using 7B LLaMA [1] and Alpaca [2] models on Anthropic’s Helpful and Harmless (HH) [3] dataset. We evaluate the results by perplexity (PPL) and reward model scores (Reward). 
With a much simplier training paradigm, we found that RRHF perform comparable result with PPO in terms of generation fluency (PPL) and alignements (Reward).

| Models | Setting | PPL       | Reward    |
| ------ | ------- | --------- | --------- |
| LLaMA  | PPO     | 42.53     | -1.62     |
| Alpaca | PPO     | **13.84** | *-1.03*   |
| LLaMA  | RRHF    | 67.12     | -1.34     |
| Alpaca | RRHF    | *14.75*   | **-1.02** |


For details, please refer to our [paper](./assets/rrhf.pdf) on [Arxiv](https://arxiv.org/pdf/2304.05302v1.pdf). RRHF is still working in progress, and there are still limitations in this preliminary study.
Due to the large cost of human evaluation, we experiment on the HH datasets and use a trained reward model *Dahoas/gptj-rm-static* trained by [Dahoas](https://github.com/Dahoas/reward-modeling.git). 
The reward model plays a role of a synthetic human feedback and the experiments is a proof-of-concept for RRHF.
We are open to any suggestions and discussions and feel free to contact us through yuanzheng.yuanzhen@alibaba-inc.com, yuanhy20@mails.tsinghua.edu.cn or chuanqi.tcq@alibaba-inc.com.

## Setting Up Environment

To set up, you can use the following command lines to set up python3.8 and pytorch requirements:
```bash
conda create -n rrhf python=3.8
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
```
Then install Hugging Face's transformers from the github repo for LLaMA models. 
```bash
git clone https://github.com/huggingface/transformers.git
pip install -e ./transformers
``` 
Install other packages:
```bash
pip install -r requirements.txt
``` 

## Train Alpaca with RRHF on Helpful and Harmless dataset
We use Helpful and Harmless dataset to compare PPO and RRHF. We use trained reward function from [Dahoas/gptj-rm-static](https://huggingface.co/Dahoas/gptj-rm-static).

| Models         | Initial Checkpoint | Sampling Models         | Reward Score | 
| -------------- | ------------------ | ----------------------- | ------------ | 
| Alpaca-7B-RRHF      | Alpaca-7B          | Alpaca-7B, responses from HH dataset  | Dahoas/gptj-rm-static      |

### Data Generation

RRHF firstly samples responses for each query in the training data from the initial models, and then scores each response (including the 'chosen' and 'rejected' response in orginal HH labels) using the reward models.

The scripts for data generation are in [./data_generation](./data_generation), you can use throught the command line:
```bash
cd ./data_generation/
bash response_gen.sh <path_to_alpaca/hf_llama_directory> <path_to_data_directory>
```

We also release our generated data for the ease of RRHF training implementation through [this link](https://drive.google.com/file/d/1nAfBt0ldSy7m5O-Sgt05SQ1rK__NmC2Z/view?usp=sharing). After download, place it to <path_to_data_directory>.

### Training with RRHF

You can train your own model with generated or released datasets using the script [train.sh](./train.sh), please note that the training process requires 8*A100 80GB GPUs, bf16 and FSDP.
In the future, we will try efficient training methods such as LoRA or Prefix-tuning or Adapter to lower the computational resource requirements.

```bash
bash ./train.sh <path_to_alpaca_directory> <save_path_directory> <path_to_data_json>
```

## Wombat: build your own chatbot.

### Introduction

To produce a more general purpose language model chatbot, we introduce **Wombat** to the model zoo of open-resourced language models. We are reseraching on how to realsese the model weights now.

| Models         | Initial Checkpoint | Sampling Models         | Reward Score | Delta Weights |
| -------------- | ------------------ | ----------------------- | ------------ | ------------- |
| Wombat-7B      | Alpaca-7B          | ChatGPT, LLaMA, Alpaca  | ChatGPT      |[GanjinZero/wombat-7b-delta](https://huggingface.co/GanjinZero/wombat-7b-delta)|
| Wombat-7B-GPT4 | Alpaca-7B          | GPT-4, GPT-3.5, OPT-IML | GPT-4        |[GanjinZero/wombat-7b-gpt4-delta](https://huggingface.co/GanjinZero/wombat-7b-gpt4-delta)|

### Weights
You should obtain LLaMa weights follow [link](https://huggingface.co/docs/transformers/main/model_doc/llama).
And you can use our provided scripts **recover_wombat_7b.sh** to recover origin Wombat weights. 

### Data and Training

- We reuse the query from Alpaca training data, and sample responses from Alpaca, LLaMA, chatGPT and text-davinci-003.
We acquire quality assessments of responses from chatGPT, and train Alpace with RRHF to become a Wombat-7B.
You can acquire the data with rewards for Wombat-7B from [this link](https://drive.google.com/file/d/1_A0Fewde9Glbm4pzdYsGI8w0df5tHEG7/view?usp=sharing), and start training your own "**Wombat**". 
Use the following command lines:

```bash
bash ./train_wombat7b.sh <path_to_alpaca_directory> <save_path_directory> <path_to_data_json>
```

- You can acquire the data with rewards for Wombat-7B-GPT4 from [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/comparision_data.json), we direct use their data trained for reward model as our training data. To use it, please first convert the data format by **clean_gpt4_compare.py**.
Use the following command lines to train Wombat-7B-GPT4:

```bash
bash ./train_wombat7b_gpt4.sh <path_to_alpaca_directory> <save_path_directory> <path_to_data_json>
```

### Responses generated by Wombat family

| Query     | Wombat-7B | Wombat-7B-GPT4          | 
| --------- | --------- | ----------------------- | 
| Give an introduction of wombat | Wombat is a marsupial animal that is native to Australia, New Guinea, and nearby islands. It is known for its distinctive appearance, with a short, stocky body, a broad head, and a short tail. Wombats are herbivores and spend most of their time burrowing in the ground, using their sharp claws to dig tunnels and create underground burrows. They have a keen sense of smell and are known to be good diggers, often digging tunnels up to 30 meters long in search of food. Wombats are nocturnal animals and are active at night, spending the day in their burrows. They are also known for their aggressive behavior and have been known to attack humans who enter their territory. | Wombat is a marsupial mammal native to Australia. It is typically found in mountainous areas and is known for its burrowing behavior. Wombats are typically around two feet long and weigh up to 30 pounds. They have a gray coat with white on their chests, and they have thick and powerful hind legs which they use for digging. Wombats are mostly nocturnal and eat a variety of insects, leaves, and grasses.  | 
| Imagine you are a time traveler from the year 3000. What technological advancements would you tell people about? | As a time traveler from the year 3000, I have had the privilege of experiencing numerous technological advancements that have revolutionized the world. One of the most significant technological advancements that I would like to share with people from the present day is the development of artificial intelligence (AI) technology. AI technology has advanced to such an extent that it has become an integral part of our daily lives. It has been used to develop intelligent machines that can perform a wide range of tasks, from driving cars to diagnosing medical conditions. AI technology has also been used to create virtual assistants like Siri and Alexa, which can answer questions, schedule appointments, and even order groceries. Another technological advancement that I would like to highlight is the development of renewable energy sources. In the year 3000, almost all of our energy needs are met by renewable sources such as solar, wind, and hydroelectric power. This has significantly reduced our carbon footprint and has helped to mitigate the effects of climate change. Finally, I would like to mention the advancements in the field of space exploration. In the year 3000, humans have been able to explore and colonize several planets beyond our solar system. Space travel has become faster, safer, and more accessible, thanks to the development of advanced propulsion systems and the use of artificial intelligence to navigate and explore new worlds. Overall, the technological advancements that I have mentioned are just a few examples of the many innovations that have been developed over the years. As time travelers, we have been able to witness firsthand the incredible progress that humanity has made in the field of technology.| I would tell people about the amazing technological advancements of the year 3000, such as artificial intelligence, robotics, genetic engineering, and quantum computing. I would also mention the advances in transportation and communication, such as levitating cars, teleportation, and holographic communication. Additionally, I would talk about the improvements to people's quality of life, such as gene editing, cybernetic implants, and virtual reality. | 
| What if Isaac Newton had focused on biology instead of physics? | As an AI language model, I cannot speculate on what might have happened if Isaac Newton had focused on biology instead of physics. However, it is worth noting that Newton was interested in a wide range of subjects, including natural history, and he made important contributions to our understanding of the natural world, including the laws of motion and the laws of optics. | If Isaac Newton had focused on biology instead of physics, he might have made important contributions to the fields of biology and medicine. For example, he could have studied the diversity of life on Earth and made groundbreaking discoveries about the origin of species. He also could have focused on the physiology of organisms and explored the functions of various organs and systems in the body. Without Newton's groundbreaking discoveries in physics, the fields of biology and medicine may have developed differently. |

## Citation
```
@misc{yuan2023rrhf,
      title={RRHF: Rank Responses to Align Language Models with Human Feedback without tears}, 
      author={Zheng Yuan and Hongyi Yuan and Chuanqi Tan and Wei Wang and Songfang Huang and Fei Huang},
      year={2023},
      eprint={2304.05302},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgements

Our implementation and experiments are based on the codes from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) and [trlX](https://github.com/CarperAI/trlx), we appreciate their open-resourced codes and [LLaMA](https://arxiv.org/abs/2302.13971v1) to promote democratized NLP research, expecially for large lanague models.
We thank Tianhang Zhu to help collecting the data and constructive discussions.

## Ref
[1]: LLaMA: Open and Efficient Foundation Language Models. Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample. https://arxiv.org/abs/2302.13971v1

[2]: Stanford alpaca: An instruction-following llama model. Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. https://github.com/tatsu-lab/stanford_alpaca

[3]: HH: Training a helpful and harmless assistant with reinforcement learning from human feedback. Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. https://arxiv.org/abs/2204.05862

