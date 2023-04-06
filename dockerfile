FROM python:3.9-slim

WORKDIR /app

RUN pip install --upgrade setuptools
RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/* 

RUN pip install --upgrade pip

RUN git clone https://github.com/DSDanielPark/GPT-BERT-Medical-QA-Chatbot.git .

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]