import streamlit as st
import openai


openai.api_key = YOUR_API_KEY

st.set_page_config(page_title="Chat GPT API EXAMPLE", page_icon=":tada:", layout="wide")

# ---- Header ----
st.subheader(
    """
                This is Test Landing Page
             """
)
st.title("EXAMPLE")

title = st.text_input("YOU:")
response = openai.Completion.create(
    model="text-davinci-003",
    prompt=title,
    temperature=0,
    max_tokens=60,
    top_p=1,
    frequency_penalty=0.5,
    presence_penalty=0,
)
if st.button("Send"):
    st.success(response.choices[0].text)
