import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset


def convert(item: str) -> np.ndarray:
    """
    Convert a string representation of an array to a numpy array.

    Args:
        item (str): String representation of an array.

    Returns:
        np.ndarray: Numpy array converted from the string representation.
    """
    item = item.strip()
    item = item[1:-1]
    item = np.fromstring(item, sep=" ")
    return item


def get_dataset(huggingface_repo: str) -> pd.DataFrame:
    """
    Load dataset from Hugging Face repository and convert to pandas DataFrame.

    Args:
        huggingface_repo (str): Name of the Hugging Face repository.

    Returns:
        pd.DataFrame: Pandas DataFrame containing the loaded dataset.
    """
    df = load_dataset(huggingface_repo, "csv")
    df = pd.DataFrame(df["train"])
    df["Q_FFNN_embeds"] = df["Q_FFNN_embeds"].apply(convert)
    df["A_FFNN_embeds"] = df["A_FFNN_embeds"].apply(convert)

    return df


def get_bert_index(
    df: pd.DataFrame, target_columns: Union[str, List[str]]
) -> faiss.IndexFlatIP:
    """
    Build and return the FAISS index for BERT embeddings.

    Args:
        df (pd.DataFrame): DataFrame containing the BERT embeddings.
        target_columns (Union[str, List[str]]): Name or list of names of the columns containing BERT embeddings.

    Returns:
        faiss.IndexFlatIP: FAISS index for BERT embeddings.
    """
    embedded_bert = df[target_columns].tolist()
    embedded_bert = np.array(embedded_bert, dtype="float32")
    index = faiss.IndexFlatIP(embedded_bert.shape[-1])
    index.add(embedded_bert)

    return index
