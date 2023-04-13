## Model details

**Organization developing the model**  
Alibaba DAMO Academy, Tsinghua University

**Model date**  
Wombat-7B was released at 2023/04/13.
Wombat-7B-GPT4 was released in 2023/04/13.

**Model version**  
Wombat-7B & Wombat-7B-GPT4.

**Training dataset**  
The training data of Wombat-7B and Wombat-7B-GPT4 is released in the [RRHF](https://github.com/GanjinZero/RRHF) and [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) respectively.

**Model type**  
Wombat-7B and Wombat-7B-GPT4 are general-purpose instruction-following language models aligned with chatGPT or GPT4 (as proxy human preferences), fine-tuned from Alpaca models.  
We use a novel methods named RRHF (Rank Response to align Human Feedback) to fine-tune Alpaca.

**How to use**
To recover Wombats from delta parameters:  
```bash 
python apply_delta.py \
    --base ./llama-7b \
    --target ./wombat-7b \
    --delta GanjinZero/wombat-7b-delta

python apply_delta.py \
    --base ./llama-7b \
    --target ./wombat-7b-gpt4 \
    --delta GanjinZero/wombat-7b-gpt4-delta
```
where **apply_delta.py** is from [code](https://github.com/GanjinZero/RRHF/blob/main/apply_delta.py).

To inference with Wombats: Please refer to [code](https://github.com/GanjinZero/RRHF/blob/main/single_sentence_inference.py).

**Citations details**  
Please cite our paper on Arxiv:
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

**License**  
Data are licensed under the CC BY NC 4.0 license.

**Where to send questions or comments about the model**  
Questions, comments and discussions about Wombats and RRHF can be sent via the [GitHub repository](https://github.com/GanjinZero/RRHF) of the project, by opening an issue.  
or send emails to yuanzheng.yuanzhen@alibaba-inc.com, yuanhy20@mails.tsinghua.edu.cn or chuanqi.tcq@alibaba-inc.com.

**Primary intended uses**  
The primary use of Wombat-7B and Wombat-7B-GPT4 is research on learning from human feedback and is a prototype of RRHF methods.

**Primary intended users**  
The primary intended users of Wombat-7B and Wombat-7B-GPT4 are researchers in natural language processing, machine learning and artificial intelligence.

**Out-of-scope use cases**  
Wombat-7B and Wombat-7B-GPT4 are not finetuned with proxy human feedback of OpenAI chatGPT and GPT4 and are not intended for use in production systems.
Any usage must not be competing with the OpenAI API.

**More information**  
Please refer to [this](../README.md) for more information.
