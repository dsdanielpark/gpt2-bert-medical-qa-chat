import streamlit as st
import tensorflow as tf
from streamlit_chat import message
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, AutoTokenizer
from modules.chatbot.inferencer import Inferencer
from modules.chatbot.dataloader import get_bert_index, get_dataset
from modules.chatbot.config import Config as CONF
from utilfunction import find_path

# Streamlit App
st.header("GPT-BERT-Medical-QA-Chatbot")

# Load necessary models and data
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(CONF.chat_params["gpt_tok"])
medi_qa_chatGPT2 = TFGPT2LMHeadModel.from_pretrained(CONF.chat_params["tf_gpt_model"])
biobert_tokenizer = AutoTokenizer.from_pretrained(CONF.chat_params["bert_tok"])
df_qa = get_dataset(CONF.chat_params["data"])
max_answer_len = CONF.chat_params["max_answer_len"]
isEval = CONF.chat_params["isEval"]
answer_index = get_bert_index(df_qa, "A_FFNN_embeds")


# Load question extractor model
@st.cache_resource
def load_tf_model(path):
    return tf.keras.models.load_model(path)


try:
    if CONF.chat_params["runDocker"]:
        tf_q_extractor_path = find_path(
            CONF.chat_params["container_mounted_folder_path"],
            "folder",
            "question_extractor_model",
        )
        question_extractor_model_v1 = load_tf_model(tf_q_extractor_path[0])
    else:
        question_extractor_model_v1 = load_tf_model(CONF.chat_params["tf_q_extractor"])
except Exception as e:
    tf_q_extractor_path = find_path("./", "folder", "question_extractor_model")
    question_extractor_model_v1 = load_tf_model(tf_q_extractor_path[0])

# Initialize chatbot inferencer
chatbot = Inferencer(
    medi_qa_chatGPT2,
    biobert_tokenizer,
    gpt2_tokenizer,
    question_extractor_model_v1,
    df_qa,
    answer_index,
    max_answer_len,
)


# Function to get model's answer
def get_model_answer(chatbot, user_input):
    return chatbot.run(user_input, isEval)


# Function to interact with chatbot
def chatgpt(input, history):
    history = history or []
    output = get_model_answer(chatbot, input)
    history.append(output)
    return history


# Maintain user input history
history_input = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []


# Function to get user input
def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text


# Main interaction loop
user_input = get_text()

if user_input:
    output = chatgpt(user_input, history_input)
    history_input.append(output)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output[0])

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
        message(
            st.session_state["past"][i],
            is_user=True,
            key=str(i) + "_user",
            avatar_style="thumbs",
        )
