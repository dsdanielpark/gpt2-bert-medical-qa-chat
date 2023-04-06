import streamlit as st
from streamlit_chat import message
import tensorflow as tf
from transformers import GPT2Tokenizer,TFGPT2LMHeadModel, AutoTokenizer, TFAutoModel
from chatbot.inferencer import Inferencer
from chatbot.dataloader import convert, get_bert_index, get_dataset
from chatbot.config import Config as CONF
from colorama import Fore, Back, Style
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

with st.echo(code_location='below'):
   total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
   num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

   Point = namedtuple('Point', 'x y')
   data = []

   points_per_turn = total_points / num_turns

   for curr_point_num in range(total_points):
      curr_turn, i = divmod(curr_point_num, points_per_turn)
      angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
      radius = curr_point_num / total_points
      x = radius * math.cos(angle)
      y = radius * math.sin(angle)
      data.append(Point(x, y))

   st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
      .mark_circle(color='#0068c9', opacity=0.5)
      .encode(x='x:Q', y='y:Q'))

# Load the chatbot model from the config.
gpt2_tokenizer=GPT2Tokenizer.from_pretrained(CONF.chat_params['gpt_tok'])
medi_qa_chatGPT2=TFGPT2LMHeadModel.from_pretrained(CONF.chat_params['tf_gpt_model'])
biobert_tokenizer = AutoTokenizer.from_pretrained(CONF.chat_params['bert_tok'])
question_extractor_model_v1=tf.keras.models.load_model(CONF.chat_params['tf_q_extractor'])
df_qa = get_dataset(CONF.chat_params['data'])
max_answer_len = CONF.chat_params['max_answer_len']
isEval = CONF.chat_params['isEval']

# Get answer index from Answer from FFNN embedding column.
answer_index = get_bert_index(df_qa, 'A_FFNN_embeds')

# Make chatbot inference object
cahtbot = Inferencer(medi_qa_chatGPT2, biobert_tokenizer, gpt2_tokenizer, question_extractor_model_v1, df_qa, answer_index, max_answer_len)


def get_model_answer(cahtbot, user_input):
    return cahtbot.run(user_input, isEval)

def chatgpt(input, history):
    history = history or []
    s = list(sum(history, ()))
    print(s)
    s.append(input)
    input = ' '.join(s)
    output = get_model_answer(input)
    history.append((input, output))
    return history, history


# Streamlit App
st.set_page_config(
    page_title="Medi-ChatGPT",
    page_icon=":robot:"
)
st.header("GPT-BERT-Medical-QA-Chatbot")

history_input = []
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()

if user_input:
    output = chatgpt(user_input, history_input)
    history_input.append([user_input, output])
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output[0])

if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')