# GPT-BERT Medical QA Chatbot


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
- chatgpt2 pretrained model [[download]](https://huggingface.co/danielpark/medical-QA-chatGPT2-v1)
- chatgpt2 tokenizer [[download]](https://huggingface.co/danielpark/medical-QA-chatGPT2-tok-v1)
- bio redditbert pretrained model [[download]](https://huggingface.co/danielpark/medical-QA-BioRedditBERT-uncased-v1)

TensorFlow models for extracting context from QA.
I temporarily share TensorFlow model weights through my personal Google Drive.
- Q extractor [[download]](https://drive.google.com/drive/folders/1VjljBW_HXXIXoh0u2Y1anPCveQCj9vnQ?usp=share_link)
- A extractor [[download]](https://drive.google.com/drive/folders/1iZ6jCiZPqjsNOyVoHcagEf3hDC5H181j?usp=share_link)


<br>

# References
[1] https://arxiv.org/abs/1706.03762 <br>
[2] https://arxiv.org/abs/1810.04805 <br>
[3] https://arxiv.org/ftp/arxiv/papers/1901/1901.08746.pdf <br>
[4] https://d4mucfpksywv.cloudfront.net/better-language-models/languagemodels.pdf%C2%A0 <br>
[5] https://github.com/ash3n/DocProduct#start-of-content <br>
[6] https://appliedaicourse.com <br>
[7] https://suniljammalamadaka.medium.com/medical-chatbot-using-bert-and-gpt2-62f0c973162f <br>
[8] Dataset https://github.com/LasseRegin/medical-question-answer-data <br>
[9] Hugging Face GPT2 https://huggingface.co/gpt2 <br>
[10] Stream it for demo https://github.com/AI-Yash/st-chat <br>
[11] Streamit https://streamlit.io/ <br>

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
