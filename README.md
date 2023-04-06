# GPT-BERT Medical QA Chatbot
![]()

<br>

# Quick Start
```
git clone https://github.com/DSDanielPark/GPT-BERT-Medical-QA-Chatbot.git
cd GPT-BERT-Medical-QA-Chatbot
pip install -e .
python main.py
```

`Pre-trained model weight needed`
Downloading datasets and model weights through the Hugging Face Hub is executed, but for some TensorFlow models, you need to manually download and place them at the top of the project folder. The information for the downloadable model is as follows, and you can visit my Hugging Face repository to check it. <br>
<br>
`chatbot/config.py`
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
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2017},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
<br>

# Tips
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
[12] https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
