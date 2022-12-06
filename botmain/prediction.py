# flake8: noqa
import pandas as pd
import numpy as np
import json
import nltk
import gc
# from tensorflow import keras
import keras
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from .preprocessing import Preprocessing

# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download('omw-1.4')


class Prediction:

    # inisiasi stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # cleaning text ke tokens
    def clean_text(self, text):
        tokens = nltk.word_tokenize(text)
        # tokens = [self.stemmer.stem(word.lower()) for word in tokens]
        tokens = [word.lower() for word in tokens]
        return tokens

    # implementasi bow
    def bag_of_words(self, text, vocab):
        tokens = self.clean_text(text)
        bow = [0] * len(vocab)
        for w in tokens:
            for idx, word in enumerate(vocab):
                if word == w:
                    bow[idx] = 1
        return np.array(bow)

    # predict jawaban berdasarkan data model
    def pred_class(self, text, vocab, labels):
        model = keras.models.load_model("dataset/data_model_extended")
        bow = self.bag_of_words(text, vocab)
        result = model.predict(np.array([bow]))[0]
        thresh = 0.235
        y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
        y_pred.sort(key=lambda x: x[1], reverse=True)

        return_list = []
        for r in y_pred:
            return_list.append(labels[r[0]])

        return return_list

    # respon jawaban yang akan ditampilkan (sementara hanya satu jawaban)
    def get_response(self, intents_list, intents_json):
        success = 0
        if len(intents_list) == 0:
            result = """
            Mohon maaf! pertanyaan yang anda masukan tidak ada jawabannya
            Silahkan gunakan contoh pertanyaan dibawah ini:
            1. Cara pakai (isi nama produk)
            2. Kegunaan (isi nama produk)
            3. Treatment (isi nama treatment)
            4. Manfaat (isi nama treatment)
            5. Harga (isi nama produk/treatment)
            """
        else:
            success = 1

            tag = intents_list[0]
            list_of_intents = intents_json["data"]

            for i in list_of_intents:
                if i["tag"] == tag:
                    print(i["id"])
                    print(i["question_category"])
                    print(i["tag"])
                    result = str(i["Jawaban"])
                    break    

        return result, success

    # jawaban berdasarkan inputan pertanyaan
    def get_result(self, message):
        words = []
        classes = []
        data_file = open("dataset/corpus_chatbot_apotek_extended.json").read()
        data = json.loads(data_file)

        for intent in data["data"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                words.extend(tokens)

            if intent["tag"] not in classes:
                classes.append(intent["tag"])

        # words = [self.stemmer.stem(word.lower()) for word in words]
        words = [word.lower() for word in words]
        words = sorted(set(words))
        classes = sorted(set(classes))

        intents = self.pred_class(message, words, classes)
        result, success = self.get_response(intents, data)
        return result, success

class Prediction_manual:

    def count_similarity(self, op, p, q):
        pre_processing = Preprocessing()
        p = pre_processing.clean_text(str(op))
        token_p = nltk.word_tokenize(p)
        token_p = list(dict.fromkeys(token_p))
        # token_p = list(dict.fromkeys(p))
        # token_p = p
        token_q = nltk.word_tokenize(q)
        token_q = list(dict.fromkeys(token_q))
        count = sum(f in token_p for f in token_q)
        return count

    def get_answer(self, q):
        data_file = open("dataset/corpus_chatbot_apotek_extended.json").read()
        data = json.loads(data_file)
        df = pd.read_json("dataset/corpus_chatbot_apotek_extended.json", orient="table")
        df["temp_question"] = q 
        df["similar_answer_by_count_words"] = df.apply(lambda x:self.count_similarity(x["optional_patterns"], x["patterns"], x["temp_question"]), axis=1)
        df = df.sort_values(by=["similar_answer_by_count_words"], ascending=False)
        df = df.reset_index()
        # print(df[["tag", "temp_question", "similar_answer_by_count_words"]].head())
        print(str(df["id"][0]))
        print(str(df["question_category"][0]))
        print(str(df["tag"][0]))
        res = ""
        res = str(df["Jawaban"][0])
        del df
        gc.collect()
        return res

