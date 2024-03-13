Development Status :: 3 - Alpha <br>
*Copyright (c) 2023 MinWoo Park*
<br>

# GPT-BERT Medical QA Chatbot
[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v2.0%20adopted-black.svg)](code_of_conduct.md)
[![Python Version](https://img.shields.io/badge/python-3.6%2C3.7%2C3.8-black.svg)](code_of_conduct.md)
![Code convention](https://img.shields.io/badge/code%20convention-pep8-black)
![Black Fomatter](https://img.shields.io/badge/code%20style-black-000000.svg)

> **Be careful when cloning this repository**: It contains large NLP model weight. (>0.45GB, [`git-lfs`](https://git-lfs.com/)) <br>
> If you want to clone without git-lfs, use this command before `git clone`. *The bandwidth provided by git-lfs for free is only 1GB per month, so there is almost no chance that a 0.45GB git-lfs download will work. So please download it manually.*
```
git lfs install --skip-smudge &
export GIT_LFS_SKIP_SMUDGE=1
```

[](https://github.com/DSDanielPark/medical-qa-bert-chatgpt/blob/main/assets/imgs/medichatbot_walle.png)

Since the advent of Chat GPT-4, there have been significant changes in the field. Nevertheless, Chat GPT-2 and Chat GPT-3 continue to be effective in specific domains as large-scale auto-regressive natural language processing models. This repository aims to qualitatively compare the performance of Chat GPT-2 and Chat GPT-4 in the medical domain, and estimate the resources and costs needed for Chat GPT-2 fine-tuning to reach the performance level of Chat GPT-4. Additionally, it seeks to assess how well up-to-date information can be incorporated and applied.

Although a few years behind GPT-4, the ultimate goal of this repository is to minimize costs and resources required for updating and obtaining usable weights after acquiring them. We plan to design experiments for few-shot learning in large-scale natural language processing models and test existing research. Please note that this repository is intended for research and practice purposes only, and we do not assume responsibility for any usage.

Additionally, this repository ultimately aims to achieve similar qualitative and quantitative performance as GPT-4 in certain domain areas through model lightweighting and optimization. For more details, please refer to my technical blog.

*Keywords: GPT-2, Streamlit, Vector DB, Medical*

<br><br><br><br><br><br>

# Contents
- [GPT-BERT Medical QA Chatbot](#gpt-bert-medical-qa-chatbot)
- [Contents](#contents)
- [Quick Start](#quick-start)
  - [Command-Line Interface](#command-line-interface)
  - [Streamlit application](#streamlit-application)
- [Docker](#docker)
  - [Build from Docker Image](#build-from-docker-image)
  - [Build from Docker Compose](#build-from-docker-compose)
  - [Build from Docker Hub](#build-from-docker-hub)
  - [Pre-trained model infomation](#pre-trained-model-infomation)
- [Dataset](#dataset)
- [Pretrained Models](#pretrained-models)
- [Cites](#cites)
- [How to cite this project](#how-to-cite-this-project)
- [Tips](#tips)
  - [About data handling](#about-data-handling)
  - [About Tensorflow-GPU handling](#about-tensorflow-gpu-handling)
  - [Remark](#remark)
- [References](#references)

<br><br><br><br><br><br>




<br>

# Quick Start
## Command-Line Interface
You can chat with the chatbot through the command-line interface using the following command.
![](https://github.com/DSDanielPark/medical-qa-bert-chatgpt/blob/main/assets/imgs/medichatbot.gif)
```
git clone https://github.com/DSDanielPark/medical-qa-bert-chatgpt.git
cd medical-qa-bert-chatgpt
pip install -e .
python main.py
```
![](https://github.com/DSDanielPark/medical-qa-bert-chatgpt/blob/main/assets/imgs/medichatbot.png)

<br>

## Streamlit application
A simple application can be implemented with streamlit as follows: <br>
![](https://github.com/DSDanielPark/medical-qa-bert-chatgpt/blob/main/assets/imgs/streamlit_app2.gif)
```
git clone https://github.com/DSDanielPark/medical-qa-bert-chatgpt.git
cd medical-qa-bert-chatgpt
pip install -e .
streamlit run chatbot.py
```
<!-- ![](https://github.com/DSDanielPark/medical-qa-bert-chatgpt/blob/main/assets/imgs/streamlit3.png) -->

# Docker
Check Docker Hub: https://hub.docker.com/r/parkminwoo91/medical-chatgpt-streamlit-v1 <br>
Docker version 20.10.24, build 297e128

## Build from Docker Image
```
git clone https://github.com/DSDanielPark/medical-qa-bert-chatgpt.git
cd medical-qa-bert-chatgpt
docker build -t chatgpt .
docker run -p 8501:8501 -v ${PWD}/:/usr/src/app/data chatgpt     # There is no cost to pay for git-lfs, just download and mount it.
```
##### Since git clone downloads what needs to be downloaded from git-lfs, the volume must be mounted as follows. Or modify `chatbot/config.py` to mount to a different folder.

## Build from Docker Compose
You can also implement it in a docker container like this: <br>
![](https://github.com/DSDanielPark/medical-qa-bert-chatgpt/blob/main/assets/imgs/docker_build.gif)
```
git clone https://github.com/DSDanielPark/medical-qa-bert-chatgpt.git
cd medical-qa-bert-chatgpt

docker compose up
```

## Build from Docker Hub

```
docker pull parkminwoo91/medical-chatgpt-streamlit-v1:latest
docker compose up
```
http://localhost:8501/

###### Streamlit is very convenient and quick to view landing pages, but lacks design flexibility and lacks control over the application layout. Also, if your application or data set is large, the entire source code will be re-run on every new change or interaction, so application flow can cause speed issues. That landing page will be replaced by flask with further optimizations. Streamlit chatbot has been recently developed, so it seems difficult to have the meaning of a simple demo now.

## Pre-trained model infomation
`Pre-trained model weight needed`
Downloading datasets and model weights through the Hugging Face Hub is executed, but for some TensorFlow models, you need to manually download and place them at the top of the project folder. The information for the downloadable model is as follows, and you can visit my Hugging Face repository to check it. <br>
<br>
`modules/chatbot/config.py`
```python
class Config:
    chat_params = {"gpt_tok":"danielpark/medical-QA-chatGPT2-tok-v1",
                   "tf_gpt_model":"danielpark/medical-QA-chatGPT2-v1",
                   "bert_tok":"danielpark/medical-QA-BioRedditBERT-uncased-v1",
                   "tf_q_extractor": "question_extractor_model",
                   "data":"danielpark/MQuAD-v1",
                   "max_answer_len": 20,
                   "isEval": False,
                   "runDocker":True, # Exceeds the bandwidth of git-lfs, mounts to local storage to find folder location for free use. I use the python utifunction package.
                   "container_mounted_folder_path": "/usr/src/app/data"} 
```

<br>

# Dataset
The Medical Question and Answering dataset(MQuAD) has been refined, including the following datasets. You can download it through the Hugging Face dataset. Use the DATASETS method as follows. You can find more infomation at [here.](https://huggingface.co/datasets/danielpark/MQuAD-v1)

```python
from datasets import load_dataset
dataset = load_dataset("danielpark/MQuAD-v1")
```

Medical Q/A datasets gathered from the following websites.
- eHealth Forum
- iCliniq
- Question Doctors
- WebMD
Data was gathered at the 5th of May 2017.

<br>

# Pretrained Models
Hugging face pretrained models
- GPT2 pretrained model [[download]](https://huggingface.co/danielpark/medical-QA-chatGPT2-v1)
- GPT2 tokenizer [[download]](https://huggingface.co/danielpark/medical-QA-chatGPT2-tok-v1)
- BIO Reddit BERT pretrained model [[download]](https://huggingface.co/danielpark/medical-QA-BioRedditBERT-uncased-v1)

TensorFlow models for extracting context from QA.
I temporarily share TensorFlow model weights through my personal Google Drive.
- Q extractor [[download]](https://drive.google.com/drive/folders/1VjljBW_HXXIXoh0u2Y1anPCveQCj9vnQ?usp=share_link)
- A extractor [[download]](https://drive.google.com/drive/folders/1iZ6jCiZPqjsNOyVoHcagEf3hDC5H181j?usp=share_link)


<br>

# Cites
```BibTex
@misc {hf_canonical_model_maintainers_2022,
        author       = { {HF Canonical Model Maintainers} },
        title        = { gpt2 (Revision 909a290) },
        year         = 2022,
        url          = { https://huggingface.co/gpt2 },
        doi          = { 10.57967/hf/0039 },
        publisher    = { Hugging Face }
}

@misc{vaswani2017attention,
      title = {Attention Is All You Need}, 
      author = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year = {2017},
      eprint = {1706.03762},
      archivePrefix = {arXiv},
      primaryClass = {cs.CL}
}
```
<br>


# How to cite this project
```BibTex
@misc{medical_qa_bert_chatgpt,
      title  = {Medical QA Bert Chat GPT}, 
      author = {Minwoo Park},
      year   = {2023},
      url    = {https://github.com/dsdanielpark/medical-qa-bert-chatgpt},
}
```


<br>

# Tips

## About data handling
The MQuAD provides embedded question and answer arrays in string format, so it is recommended to convert the string-formatted arrays into float format as follows. This measure has been applied to save resources and time used for embedding.

```python
from datasets import load_dataset
from utilfunction import col_convert
import pandas as pd

qa = load_dataset("danielpark/MQuAD-v1", "csv")
df_qa = pd.DataFrame(qa['train'])
df_qa = col_convert(df_qa, ['Q_FFNN_embeds', 'A_FFNN_embeds'])
```

## About Tensorflow-GPU handling
Since the nvidia GPU driver fully supports wsl2, the method of supporting TensorFlow's gpu has changed. Please refer to the following pages to install it.
- https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- https://www.tensorflow.org/install/pip?hl=ko

<br>

## Remark
I have trained the model for 2 epochs using the mentioned dataset, utilizing 40 computing units from Google Colab Pro. The training was conducted for about 12 hours using an A100 multi-GPU with 56 GB of RAM or more. In the case of relatively simple question extractor or answer extractor models that perform summarization and indexing, the time required for training is minimal, and they are included in the inference module to evaluate whether the learning has been carried out appropriately. If the model is only responding to simple questions, the inference module should be changed; 
however, it is currently included in the evaluation unnecessarily to check performance and calculate the time and resources consumed. I plan to update this information once sufficient training is completed (by incorporating additional datasets), or when funding for experiments and resources to derive adequate learning. <br>

- Training 2 Epoch with `MQuAD` dataset, Comsuming 40 Google Colab Pro Computing unit, Take 12 hours using an A100 multi-GPU with 56 GB of RAM or more.

<br>

# References
1. [Paper: Attention is All You Need](https://arxiv.org/abs/1706.03762)
2. [Paper: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
3. [Paper: GPT-2: Language Models are Unsupervised Multitask Learners](https://arxiv.org/ftp/arxiv/papers/1901/1901.08746.pdf)
4. [Paper: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/languagemodels.pdf%C2%A0)
5. [GitHub Repository: DocProduct](https://github.com/ash3n/DocProduct#start-of-content)
6. [Applied AI Course](https://appliedaicourse.com)
7. [Medium Article: Medical Chatbot using BERT and GPT-2](https://suniljammalamadaka.medium.com/medical-chatbot-using-bert-and-gpt2-62f0c973162f)
8. [GitHub Repository: Medical Question Answer Data](https://github.com/LasseRegin/medical-question-answer-data)
9. [Hugging Face Model Hub: GPT-2](https://huggingface.co/gpt2)
10. [GitHub Repository: Streamlit Chat](https://github.com/AI-Yash/st-chat)
11. [Streamlit Documentation](https://streamlit.io/)
12. [Streamlit Tutorial: Deploying Streamlit Apps with Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
13. [ChatterBot Documentation](https://chatterbot.readthedocs.io/en/stable/logic/index.html)
14. [Blog Post: 3 Steps to Fix App Memory Leaks](https://blog.streamlit.io/3-steps-to-fix-app-memory-leaks/)
15. [Blog Post: Common App Problems & Resource Limits](https://blog.streamlit.io/common-app-problems-resource-limits/)
16. [GitHub Gist: Streamlit Chatbot Example](https://gist.github.com/DSDanielPark/5d34b2f53709a7007b0d3a5e9f23c0a6) (Lightweight and optimized)
17. [Databricks Blog: Democratizing Magic: ChatGPT and Open Models](https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html)
18. [GitHub Repository: Pyllama](https://github.com/juncongmoo/pyllama)