import re
from modules.chatbot.const import CONTRACTIONS


def decontracted(phrase):
    """
    Decontract a phrase.

    Args:
        phrase (str): The input phrase.

    Returns:
        str: Decontracted phrase.
    """
    for key, value in CONTRACTIONS.items():
        phrase = phrase.replace(key, value)
    return phrase


def preprocess(text):
    """
    Preprocess text.

    Args:
        text (str): The input text.

    Returns:
        str: Preprocessed text.
    """
    text = text.lower()
    text = decontracted(text)
    text = re.sub(r"[$)\?\"’.°!;'€%:,(/]", "", text)
    text = re.sub(r"\u200b|\xa0|-", " ", text)
    return text
