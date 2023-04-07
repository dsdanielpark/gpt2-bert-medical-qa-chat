# GPT-BERT Medical QA Chatbot
[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v2.0%20adopted-black.svg)](code_of_conduct.md)
[![Python Version](https://img.shields.io/badge/python-3.6%2C3.7%2C3.8-black.svg)](code_of_conduct.md)
![Code convention](https://img.shields.io/badge/code%20convention-pep8-black)
![Black Fomatter](https://img.shields.io/badge/code%20style-black-000000.svg)

> **Be careful when cloning this repository**: It contains large NLP model weight. (>0.45GB, [`git-lfs`](https://git-lfs.com/)) <br>
> If you want to clone without git-lfs, use this command before `git clone`.
```
git lfs install --skip-smudge &
export GIT_LFS_SKIP_SMUDGE=1
```
 
Since the advent of Chat GPT-4, there have been significant changes in the field. Nevertheless, Chat GPT-2 and Chat GPT-3 continue to be effective in specific domains as large-scale auto-regressive natural language processing models. This repository aims to qualitatively compare the performance of Chat GPT-2 and Chat GPT-4 in the medical domain, and estimate the resources and costs needed for Chat GPT-2 fine-tuning to reach the performance level of Chat GPT-4. Additionally, it seeks to assess how well up-to-date information can be incorporated and applied.

Although a few years behind GPT-4, the ultimate goal of this repository is to minimize costs and resources required for updating and obtaining usable weights after acquiring them. We plan to design experiments for few-shot learning in large-scale natural language processing models and test existing research. Please note that this repository is intended for research and practice purposes only, and we do not assume responsibility for any usage.

Additionally, this repository ultimately aims to achieve similar qualitative and quantitative performance as GPT-4 in certain domain areas through model lightweighting and optimization. For more details, please refer to my technical blog.

![](https://github.com/DSDanielPark/GPT-BERT-Medical-QA-Chatbot/blob/main/asset/medichatbot.gif)

<br>

# Quick Start
You can chat with the chatbot through the command-line interface using the following command.
```
git clone https://github.com/DSDanielPark/GPT-BERT-Medical-QA-Chatbot.git
cd GPT-BERT-Medical-QA-Chatbot
pip install -e .
python main.py
```

![](https://github.com/DSDanielPark/GPT-BERT-Medical-QA-Chatbot/blob/main/asset/medichatbot.png)

<br>

`Pre-trained model weight needed`
Downloading datasets and model weights through the Hugging Face Hub is executed, but for some TensorFlow models, you need to manually download and place them at the top of the project folder. The information for the downloadable model is as follows, and you can visit my Hugging Face repository to check it. <br>
<br>
`modules/chatbot/config.py`
```python
class Config:
    chat_params = {"gpt_tok":"danielpark/medical-QA-chatGPT2-tok-v1",
                   "tf_gpt_model":"danielpark/medical-QA-chatGPT2-v1",
                   "bert_tok":"danielpark/medical-QA-BioRedditBERT-uncased-v1",
                   "tf_q_extractor": "question_extractor_model_v1",
                   "data":"danielpark/MQuAD-v1",
                   "max_answer_len": 20,
                   "isEval": False}
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

# Tips

## About data handling
The MQuAD provides embedded question and answer arrays in string format, so it is recommended to convert the string-formatted arrays into float format as follows. This measure has been applied to save resources and time used for embedding.

```python
from datasets import load_dataset
import pandas as pd
import numpy as np

qa = load_dataset("danielpark/MQuAD-v1", "csv")
qa = pd.DataFrame(qa['train'])

def convert(item):
    item = item.strip()  
    item = item[1:-1]   
    item = np.fromstring(item, sep=' ') 
    return item

qa['Q_FFNN_embeds'] = qa['Q_FFNN_embeds'].apply(convert)
qa['A_FFNN_embeds'] = qa['A_FFNN_embeds'].apply(convert)
```

## About Tensorflow-GPU handling
Since the nvidia GPU driver fully supports wsl2, the method of supporting TensorFlow's gpu has changed. Please refer to the following pages to install it.
- https://docs.nvidia.com/cuda/wsl-user-guide/index.html
- https://www.tensorflow.org/install/pip?hl=ko

<br>

# References
[1] https://arxiv.org/abs/1706.03762 <br>
[2] https://arxiv.org/abs/1810.04805 <br>
[3] https://arxiv.org/ftp/arxiv/papers/1901/1901.08746.pdf <br>
[4] https://d4mucfpksywv.cloudfront.net/better-language-models/languagemodels.pdf%C2%A0 <br>
[5] https://github.com/ash3n/DocProduct#start-of-content <br>
[6] https://appliedaicourse.com <br>
[7] https://suniljammalamadaka.medium.com/medical-chatbot-using-bert-and-gpt2-62f0c973162f <br>
[8] https://github.com/LasseRegin/medical-question-answer-data <br>
[9] https://huggingface.co/gpt2 <br>
[10] https://github.com/AI-Yash/st-chat <br>
[11] https://streamlit.io/ <br>
[12] https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker <br>
[13] https://chatterbot.readthedocs.io/en/stable/logic/index.html
