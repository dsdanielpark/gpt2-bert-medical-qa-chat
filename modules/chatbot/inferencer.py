import numpy as np
import tensorflow as tf
from typing import List
from nltk.translate.bleu_score import sentence_bleu
from modules.chatbot.preprocessor import preprocess


class Inferencer:
    def __init__(
        self,
        medical_qa_gpt_model: tf.keras.Model,
        bert_tokenizer: tf.keras.preprocessing.text.Tokenizer,
        gpt_tokenizer: tf.keras.preprocessing.text.Tokenizer,
        question_extractor_model: tf.keras.Model,
        df_qa: pd.DataFrame,
        answer_index: faiss.IndexFlatIP,
        answer_len: int,
    ) -> None:
        """
        Initialize Inferencer with necessary components.

        Args:
            medical_qa_gpt_model (tf.keras.Model): Medical Q&A GPT model.
            bert_tokenizer (tf.keras.preprocessing.text.Tokenizer): BERT tokenizer.
            gpt_tokenizer (tf.keras.preprocessing.text.Tokenizer): GPT tokenizer.
            question_extractor_model (tf.keras.Model): Question extractor model.
            df_qa (pd.DataFrame): DataFrame containing Q&A pairs.
            answer_index (faiss.IndexFlatIP): FAISS index for answers.
            answer_len (int): Length of the answer.
        """
        self.biobert_tokenizer = bert_tokenizer
        self.question_extractor_model = question_extractor_model
        self.answer_index = answer_index
        self.gpt_tokenizer = gpt_tokenizer
        self.medical_qa_gpt_model = medical_qa_gpt_model
        self.df_qa = df_qa
        self.answer_len = answer_len

    def get_gpt_inference_data(
        self, question: str, question_embedding: np.ndarray
    ) -> List[int]:
        """
        Get GPT inference data.

        Args:
            question (str): Input question.
            question_embedding (np.ndarray): Embedding of the question.

        Returns:
            List[int]: GPT inference data.
        """
        topk = 20
        scores, indices = self.answer_index.search(
            question_embedding.astype("float32"), topk
        )
        q_sub = self.df_qa.iloc[indices.reshape(20)]
        line = "`QUESTION: %s `ANSWER: " % (question)
        encoded_len = len(self.gpt_tokenizer.encode(line))
        for i in q_sub.iterrows():
            line = (
                "`QUESTION: %s `ANSWER: %s " % (i[1]["question"], i[1]["answer"]) + line
            )
            line = line.replace("\n", "")
            encoded_len = len(self.gpt_tokenizer.encode(line))
            if encoded_len >= 1024:
                break
        return self.gpt_tokenizer.encode(line)[-1024:]

    def get_gpt_answer(self, question: str, answer_len: int) -> str:
        """
        Get GPT answer.

        Args:
            question (str): Input question.
            answer_len (int): Length of the answer.

        Returns:
            str: GPT generated answer.
        """
        preprocessed_question = preprocess(question)
        truncated_question = (
            " ".join(preprocessed_question.split(" ")[:500])
            if len(preprocessed_question.split(" ")) > 500
            else preprocessed_question
        )
        encoded_question = self.biobert_tokenizer.encode(truncated_question)
        padded_question = tf.keras.preprocessing.sequence.pad_sequences(
            [encoded_question], maxlen=512, padding="post"
        )
        question_mask = np.where(padded_question != 0, 1, 0)
        embeddings = self.question_extractor_model(
            {"question": padded_question, "question_mask": question_mask}
        )
        gpt_input = self.get_gpt_inference_data(truncated_question, embeddings.numpy())
        mask_start = len(gpt_input) - list(gpt_input[::-1]).index(4600) + 1
        input = gpt_input[: mask_start + 1]
        if len(input) > (1024 - answer_len):
            input = input[-(1024 - answer_len) :]
        gpt2_output = self.gpt_tokenizer.decode(
            self.medical_qa_gpt_model.generate(
                input_ids=tf.constant([np.array(input)]),
                max_length=1024,
                temperature=0.7,
            )[0]
        )
        answer = gpt2_output.rindex("`ANSWER: ")
        return gpt2_output[answer + len("`ANSWER: ") :]

    def inf_func(self, question: str) -> str:
        """
        Run inference for the given question.

        Args:
            question (str): Input question.

        Returns:
            str: Generated answer.
        """
        answer_len = self.answer_len
        return self.get_gpt_answer(question, answer_len)

    def eval_func(self, question: str, answer: str) -> float:
        """
        Evaluate generated answer against ground truth.

        Args:
            question (str): Input question.
            answer (str): Generated answer.

        Returns:
            float: BLEU score.
        """
        answer_len = 20
        generated_answer = self.get_gpt_answer(question, answer_len)
        reference = [answer.split(" ")]
        candidate = generated_answer.split(" ")
        score = sentence_bleu(reference, candidate)
        return score

    def run(self, question: str, isEval: bool) -> str:
        """
        Run inference for the given question.

        Args:
            question (str): Input question.
            isEval (bool): Whether to evaluate or not.

        Returns:
            str: Generated answer.
        """
        answer = self.inf_func(question)
        if isEval:
            bleu_score = self.eval_func(question, answer)
            print(f"The sentence_bleu score is {bleu_score}")
        return answer
