from datasets import load_dataset
import pandas as pd
import numpy as np
import faiss


def convert(item):
    item = item.strip()  
    item = item[1:-1]   
    item = np.fromstring(item, sep=' ') 
    return item


def get_dataset(huggingface_repo):
    df = load_dataset(huggingface_repo, "csv")
    df = pd.DataFrame(df['train'])
    df['Q_FFNN_embeds'] = df['Q_FFNN_embeds'].apply(convert)
    df['A_FFNN_embeds'] = df['A_FFNN_embeds'].apply(convert)

    return df


def get_bert_index(df, target_columns):
    embedded_bert = df[target_columns].tolist()
    embedded_bert = np.array(embedded_bert)
    embedded_bert = embedded_bert.astype('float32')
    indexs = faiss.IndexFlatIP(embedded_bert.shape[-1])
    indexs.add(embedded_bert)

    return indexs