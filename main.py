import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, AutoTokenizer, TFAutoModel
from modules.chatbot.inferencer import Inferencer
from modules.chatbot.dataloader import convert, get_bert_index, get_dataset
from modules.chatbot.config import Config as CONF
from colorama import Fore, Back, Style
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)


def main():
    # Load the chatbot model from the config.
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(CONF.chat_params["gpt_tok"])
    medi_qa_chatGPT2 = TFGPT2LMHeadModel.from_pretrained(
        CONF.chat_params["tf_gpt_model"]
    )
    biobert_tokenizer = AutoTokenizer.from_pretrained(CONF.chat_params["bert_tok"])
    try:
        question_extractor_model_v1 = tf.keras.models.load_model(
            CONF.chat_params["tf_q_extractor"]
        )
    except Exception as e:
        print(e)

    df_qa = get_dataset(CONF.chat_params["data"])
    max_answer_len = CONF.chat_params["max_answer_len"]
    isEval = CONF.chat_params["isEval"]

    # Get answer index from Answer from FFNN embedding column.
    answer_index = get_bert_index(df_qa, "A_FFNN_embeds")

    # Make chatbot inference object
    cahtbot = Inferencer(
        medi_qa_chatGPT2,
        biobert_tokenizer,
        gpt2_tokenizer,
        question_extractor_model_v1,
        df_qa,
        answer_index,
        max_answer_len,
    )

    # Start chatbot
    print("========================================")
    print(Back.BLUE + "          Welcome to MediChatBot        " + Back.RESET)
    print("========================================")
    print("If you enter quit, q, stop, chat will be ended.")
    print(
        "MediChatBot v1 is not an official service and is not responsible for any usage."
    )
    print(
        "Please enter your message below.\nThis chatbot is not sufficiently trained and the dataset is not properly cleaned, so it does not have a meaning beyond the demo version."
    )

    # Chat
    while True:
        user_input = input(Fore.BLUE + "You: " + Fore.RESET)
        if user_input.lower() in ["quit", "q", "stop"]:
            print("========================================")
            print(
                Fore.RED
                + "              Chat Ended.          "
                + Fore.RESET
                + "\n\nThank you for using DSDanielPark's chatbot. Please visit our GitHub and Hugging Face for more information. \n\n - github: https://github.com/DSDanielPark/GPT-BERT-Medical-QA-Chatbot \n - hugging-face: https://huggingface.co/datasets/danielpark/MQuAD-v1 "
            )
            print("========================================")
            break

        response = cahtbot.run(user_input, isEval)
        print(
            Fore.BLUE
            + Style.BRIGHT
            + "MediChatBot: "
            + response
            + Fore.RESET
            + Style.RESET_ALL
        )
        response = ""


if __name__ == "__main__":
    main()
