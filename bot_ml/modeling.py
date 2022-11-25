import pandas as pd
import numpy as np
import random
import re
import time
import io 
from datetime import timedelta
import json
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import gensim
from gensim.models import word2vec
import multiprocessing
# import chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

# nltk.download('punkt')

from .preprocessing import Preprocessing
from .postprocessing import Postprocessing


class Modeling_rule_based:

    # GET PRODUCT NAME MANUAL
    # parsing product name manual
    def get_product(self, c, a, v):
        v = re.sub("[^a-z0-9]+", " ", v.lower())
        # untuk treatment_info, product name didapatkan dari Version ID
        p = v
        # sedangkan untuk product_info, cek apakah Version ID itu terdiri dari satu suku kata, jika iya lakukan parsing dari data Answer
        if c == "product_info":
            if " " not in v:
                p = a
                p = p[p.find("*") : len(p)]
                p = p[: p.find("\n")]
                p = p.lower()
                p = re.sub("[^0-9a-z]+", " ", p)

                replacements = {
                    "(?<=\s|[0-9])+(g|gr)+(?=\s|\.|$)": "gram",
                    "(?<=\s|[0-9])+mg+(?=\s|\.|$)": "miligram",
                    "(?<=\s|[0-9])+ml+(?=\s|\.|$)": "mililiter",
                    "(?<=\s|[0-9])+l+(?=\s|\.|$)": "liter",
                }

                for key, value in replacements.items():
                    p = re.sub(key, value, p)

                if len(p)==0:
                    v = re.sub("[^a-z0-9]+", " ", v.lower())
                    p = v

        return p

    # CREATE QUESTION CANDIDATES MANUAL
    def create_questions(self, c, product, version):
        q = product
        # q = version
        list_q = []
        if c == "product_info":
            q1 = q
            q2 = "pakai " + q
            list_q = [q1, q2]
        elif c == "treatment_info":
            q1 = q
            q2 = "treatment " + q
            list_q = [q1, q2]

        return list_q
    
    # CREATE QUESTION CANDIDATES MANUAL #2
    def create_optional_questions(self, c, product, version):
        q = product
        # q = version
        list_q = []
        if c == "product_info":
            q1 = q
            q2 = "pakai " + q
            q3 = "kegunaan " + q
            q4 = "penyimpanan " + q
            q5 = "harga " + q
            list_q = [q1, q2, q3, q4, q5]
        elif c == "treatment_info":
            q1 = q
            q2 = "treatment " + q
            q3 = "pengerjaan " + q
            q4 = "manfaat " + q
            q5 = "harga " + q
            list_q = [q1, q2, q3, q4, q5]
        elif c == "general_info":
            q1 = q
            q2 = "apa " + q
            q3 = "informasi " + q
            q4 = "produk " + q
            q5 = "program " + q
            list_q = [q1, q2, q3, q4, q5]

        return list_q

    # CREATE QUESTION PATTERN FROM QUESTION CANDIDATES
    # custom pattern
    def create_pattern_questions(
        self, answer_tokenized, product_name, current_question
    ):
        list_all = []

        # coba-coba pattern
        # list_all.append(product_name)
        list_all = current_question
        # list_all = answer_tokenized

        # gabungan question pattern
        # for t in answer_tokenized:
        #     new_question = str(t) + " " + str(product_name)
        #     list_all.append(new_question)
        
        # list_all = current_question + list_all
        # list_all = list(dict.fromkeys(list_all))

        return list_all

class Modeling_word2vec:
    def convert_data(self):
        start_time = time.time()
        print('Streaming wiki...')
        id_wiki = gensim.corpora.WikiCorpus(
            'dataset/idwiki-latest-pages-articles.xml.bz2', dictionary={}, lower=True
        )
        
        article_count = 0
        with io.open('dataset/idwiki_new_lower.txt', 'w', encoding='utf-8') as wiki_txt:
            for text in id_wiki.get_texts():

                wiki_txt.write(" ".join(text) + '\n')
                article_count += 1

                if article_count % 10000 == 0:
                    print('{} articles processed'.format(article_count))
            print('total: {} articles'.format(article_count))

        finish_time = time.time()
        print('Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))

    def training_model(self):
        start_time = time.time()
        print('Training Word2Vec Model...')
        sentences = word2vec.LineSentence('dataset/idwiki_new_lower.txt')
        id_w2v = word2vec.Word2Vec(sentences, vector_size=200, workers=multiprocessing.cpu_count()-1)
        id_w2v.save('dataset/idwiki_word2vec_200_new_lower.model')
        finish_time = time.time()

        print('Finished. Elapsed time: {}'.format(timedelta(seconds=finish_time-start_time)))

    def testing_model(self, word):
        namaFileModel = "dataset/idwiki_word2vec_200_new_lower.model"
        model = gensim.models.Word2Vec.load(namaFileModel)
        hasil = model.wv.most_similar(word)
        # print(hasil)
        res = []
        for i in hasil:
            res.append(i[0])
        # print(res)
        return res

class Modeling_chatterbot:
    def chat_bot(self):
        # Create a new chat bot named Charlie
        chatbot = ChatBot('Charlie')

        trainer = ListTrainer(chatbot)

        trainer.train([
            "Hi, can I help you?",
            "Sure, I'd like to book a flight to Iceland.",
            "Your flight has been booked."
        ])

        # Get a response to the input text 'I would like to book a flight.'
        response = chatbot.get_response('I would like to book a flight.')

        print(response)

class Modeling_learning_based:

    # TRAINING MODEL FROM DATA CORPUS
    def training_model(self):

        words = []
        classes = []
        data_X = []
        data_y = []

        # data_file = open("dataset/corpus_chatbot_apotek.json").read()
        data_file = open("dataset/corpus_chatbot_apotek_extended.json").read()
        data = json.loads(data_file)

        for intent in data["data"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                words.extend(tokens)
                data_X.append(pattern)
                data_y.append(intent["tag"])
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        # words = [stemmer.stem(word.lower()) for word in words]
        words = [word.lower() for word in words]
        words = sorted(set(words))
        classes = sorted(set(classes))

        training = []
        out_empty = [0] * len(classes)

        for idx, doc in enumerate(data_X):
            bow = []
            # text = stemmer.stem(doc.lower())
            text = doc.lower()
            for word in words:
                bow.append(1) if word in text else bow.append(0)

            output_row = list(out_empty)
            output_row[classes.index(data_y[idx])] = 1

            training.append([bow, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)

        train_X = np.array(list(training[:, 0]))
        train_Y = np.array(list(training[:, 1]))

        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_X[0]),), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_Y[0]), activation="softmax"))

        adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        model.compile(
            loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"]
        )

        print(model.summary())
        model.fit(x=train_X, y=train_Y, epochs=150, verbose=1)

        # SAVE DATA MODEL PICKLE
        model.save("dataset/data_model_extended")

class Generate_corpus:
    def generate(self):
        # DATASET RAW PERTAMA
        # GENERATE CORPUS (ANSWERED - QUESTIONS PATTERN)
        # LOAD DATASET
        df_raw = pd.read_excel("dataset/Chatbot_node_wording.xlsx")
        ## filter data (ignore null answers, and get only product_info and treatment_info)
        df_raw = df_raw.loc[
            df_raw["Content ID"].isin(["product_info", "treatment_info"])
        ]
        df_raw = df_raw.loc[~df_raw[["Answer 1"]].isna().any(axis=1)]
        # print(len(df_raw))

        ## Process per Batch
        batch_size = 100
        pre_processing = Preprocessing()
        rule_based = Modeling_rule_based()
        df = pd.DataFrame()

        for batch_number, batch_df in df_raw.groupby(
            np.arange(len(df_raw)) // batch_size
        ):
            # PREPROCESSING DATASET
            ## get cleaned tokenized word for learning model
            batch_df["answer_tokenized"] = batch_df["Answer 1"].apply(
                lambda x: pre_processing.pre_process(x)
            )

            # MODELING CORPUS BASED ON RULE BASED
            ## training (create corpus for questions), and testing
            batch_df["product_name"] = batch_df.apply(
                lambda x: rule_based.get_product(
                    x["Content ID"], x["Answer 1"], x["Version ID"]
                ),
                axis=1,
            )
            batch_df.loc[batch_df["product_name"] == "", "product_name"] = np.nan
            batch_df.loc[batch_df["product_name"] == " ", "product_name"] = np.nan
            batch_df["product_name"].fillna(batch_df["Version ID"], inplace=True)
            batch_df["questions_candidate"] = batch_df.apply(
                lambda x: rule_based.create_questions(
                    x["Content ID"], x["product_name"], x["Version ID"]
                ),
                axis=1,
            )
            batch_df["questions_pattern"] = batch_df.apply(
                lambda x: rule_based.create_pattern_questions(
                    x["answer_tokenized"], x["product_name"], x["questions_candidate"]
                ),
                axis=1,
            )
            batch_df["questions_optional_pattern"] = batch_df.apply(
                lambda x: rule_based.create_optional_questions(
                    x["Content ID"], x["product_name"], x["Version ID"]
                ),
                axis=1,
            )
            df = pd.concat([df, batch_df], ignore_index=True)
        # print(len(df))

        # GENERATE CORPUS FROM DATASET (QUESTIONS CANDIDATE)
        ## transform into json file
        df_res = df[
            [
                "Version ID",
                "Content ID",
                "product_name",
                "questions_pattern",
                "Answer 1",
                "questions_optional_pattern"
            ]
        ]
        df_res = df_res.rename(
            columns={
                "Version ID": "id",
                "Content ID": "content",
                "product_name": "tag",
                "questions_pattern": "patterns",
                "Answer 1": "responses",
                "questions_optional_pattern": "optional_patterns"
            }
        )

        # DATASET RAW KEDUA
        df_res2 = pd.read_excel("dataset/Worksheet AI ERHA extended.xlsx")
        df_res2["content"] = "general_info"
        df_res2["Sub Kategori Pertanyaan 2"] = df_res2["Sub Kategori Pertanyaan 2"].apply(lambda x: pre_processing.clean_text(x))
        df_res2["questions_optional_pattern"] = df_res2.apply(
            lambda x: rule_based.create_optional_questions(
                x["content"], x["Sub Kategori Pertanyaan 2"], x["No"]
            ),
            axis=1,
        )

        df_res2 = df_res2[
            [
                "No",
                "content",
                "Sub Kategori Pertanyaan 2",
                "Pertanyaan",
                "Jawaban",
                "questions_optional_pattern"
            ]
        ]

        df_res2 = df_res2.rename(
            columns={
                "No": "id",
                "Sub Kategori Pertanyaan 2": "tag",
                "Pertanyaan": "patterns",
                "Jawaban": "responses",
                "questions_optional_pattern": "optional_patterns"
            }
        )
        
        # print(len(df_res))
        # print(len(df_res2))
        df_res_all = pd.concat([df_res, df_res2])
        # print(len(df_res_all))

        post_processing = Postprocessing()
        post_processing.generate_corpus(df_res_all)


class Generate_model:
    def generate(self):
        # TRAINING TO GET DATA MODEL (QUESTION ANSWER RELATION)
        modeling_ml = Modeling_learning_based()
        modeling_ml.training_model()

