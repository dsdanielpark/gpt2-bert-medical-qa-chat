class Config:
    chat_params = {
        "gpt_tok": "danielpark/medical-QA-chatGPT2-tok-v1",
        "tf_gpt_model": "danielpark/medical-QA-chatGPT2-v1",
        "bert_tok": "danielpark/medical-QA-BioRedditBERT-uncased-v1",
        "tf_q_extractor": "question_extractor_model",
        "data": "danielpark/MQuAD-v1",
        "max_answer_len": 20,
        "isEval": False,
        "runDocker": True,  # Exceeds the bandwidth of git-lfs, mounts to local storage to find folder location for free use. I use the python utifunction package.
        "container_mounted_folder_path": "/usr/src/app/data",
    }
