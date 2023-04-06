from nltk.translate.bleu_score import sentence_bleu
from chatbot.preprocessor import *
import tensorflow as tf
import numpy as np


class Inferencer:
  def __init__(self, medical_qa_gpt_model, bert_tokenizer, gpt_tokenizer, question_extractor_model, df_qa, answer_index, answer_len) -> None:
    self.biobert_tokenizer = bert_tokenizer
    self.question_extractor_model = question_extractor_model
    self.answer_index = answer_index
    self.gpt_tokenizer = gpt_tokenizer
    self.medical_qa_gpt_model = medical_qa_gpt_model
    self.df_qa = df_qa
    self.answer_len = answer_len


  def get_gpt_inference_data(self, question, question_embedding):
    topk=20
    scores,indices=self.answer_index.search(
                    question_embedding.astype('float32'), topk)
    q_sub=self.df_qa.iloc[indices.reshape(20)]
    line = '`QUESTION: %s `ANSWER: ' % (question)
    encoded_len=len(self.gpt_tokenizer.encode(line))
    for i in q_sub.iterrows():
      line='`QUESTION: %s `ANSWER: %s ' % (i[1]['question'],i[1]['answer']) + line
      line=line.replace('\n','')
      encoded_len=len(self.gpt_tokenizer.encode(line))
      if encoded_len>=1024:
        break
    return self.gpt_tokenizer.encode(line)[-1024:]

  def get_gpt_answer(self, question, answer_len):
    preprocessed_question=preprocess(question)
    question_len=len(preprocessed_question.split(' '))
    truncated_question=preprocessed_question
    if question_len>500:
      truncated_question=' '.join(preprocessed_question.split(' ')[:500])
    encoded_question= self.biobert_tokenizer.encode(truncated_question, truncation=True, max_length=1000)
    max_length=512
    padded_question=tf.keras.preprocessing.sequence.pad_sequences(
        [encoded_question], maxlen=max_length, padding='post')
    question_mask=[[1 if token!=0 else 0 for token in question] for question in padded_question]
    embeddings=self.question_extractor_model({'question':np.array(padded_question),'question_mask':np.array(question_mask)})
    gpt_input=self.get_gpt_inference_data(truncated_question, embeddings.numpy())
    mask_start = len(gpt_input) - list(gpt_input[::-1]).index(4600) + 1
    input=gpt_input[:mask_start+1]
    if len(input)>(1024-answer_len):
      input=input[-(1024-answer_len):]
    gpt2_output=self.gpt_tokenizer.decode(self.medical_qa_gpt_model.generate(input_ids=tf.constant([np.array(input)]),max_length=1024,temperature=0.7)[0])
    answer=gpt2_output.rindex('`ANSWER: ')
    return gpt2_output[answer+len('`ANSWER: '):]


  def inf_func(self, question):
    answer_len = self.answer_len
    return self.get_gpt_answer(question, answer_len)


  def eval_func(self, question, answer):
    print(f'Q for eval func: {question}')
    print(f'A for eval func: {answer}')
    answer_len=self.answer_len
    generated_answer=self.get_gpt_answer(question, answer_len)
    reference = [answer.split(' ')]
    candidate = generated_answer.split(' ')
    score = sentence_bleu(reference, candidate)
    return score
  
  def run(self, question, isEval):
    answer = self.inf_func(question)
    if isEval:
      bleu_score = self.eval_func(question, answer)
      print(f'The sentence_bleu score is {bleu_score}')

    return answer