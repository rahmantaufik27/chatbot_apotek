import pandas as pd
import numpy as np
import requests
import random
import re
import string
import json 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#import torch
#import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader
#from nltk_utils import bag_of_words 
#from model import NeuralNet 
import nltk
# nltk.download('punkt')

from preprocessing import Preprocessing

class Modeling_rule_based:

    # get product name manual
    ## filter product name, perlu di regex lagi
    def get_product(self, p):
        p = p[p.find('*'):len(p)]
        p = p[:p.find('\n')]
        p = p.lower()
        p = re.sub("[^0-9a-z]+", " ", p)

        replacements = {
            '(?<=\s|[0-9])+(g|gr)+(?=\s|\.|$)' : 'gram',
            '(?<=\s|[0-9])+mg+(?=\s|\.|$)' : 'miligram',
            '(?<=\s|[0-9])+ml+(?=\s|\.|$)' : 'mililiter',
            '(?<=\s|[0-9])+l+(?=\s|\.|$)' : 'liter'
        }

        for key, value in replacements.items():
            p = re.sub(key, value, p)
            
        return p

    # create question manual
    ## keyword berdasarkan huruf terbanyak yang muncul
    def create_questions(self, c, q):
        list_q = []
        if c == "product_info":
            q1 = q 
            q2 = "kegunaan " + q
            q3 = "penggunaan " + q
            q4 = "cara pakai " + q
            q5 = "bagaimana menggunakan " + q
            q6 = "cara menggunakan " + q
            q7 = "cara penyimpanan " + q
            q8 = "bagaimana menyimpan " + q
            q9 = "cara menyimpan " + q
            q10 = "apa itu " + q
            q11 = "harga " + q
            q12 = "berapa biaya " + q
            list_q = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12]
        elif c == "treatment_info":
            q1 = q
            q2 = "apa itu " + q
            q3 = "manfaat " + q
            q4 = "manfaat menggunakan" + q
            q5 = "keuntungan menggunakan " + q
            q6 = "prosedur " + q
            q7 = "bagaimana pengerjaan " + q
            q8 = "bagaimana cara kerja " + q
            q9 = "durasi " + q
            q10 = "berapa lama " + q
            q11 = "harga " + q
            q12 = "berapa biaya " + q
            list_q = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12]
        
        return list_q

    # create question from all pattern 
    ## custom pattern and answer_tokenized pattern
    def create_pattern_questions(self, answer_tokenized, product_name, current_question):
        list_all = []
        for t in answer_tokenized:
            new_question = str(t) + " " + str(product_name)
            list_all.append(new_question)

        list_all = current_question + list_all
        list_all = list(dict.fromkeys(list_all))

        return list_all

class Modeling_learning_based:

    # training model from json output
    def training_model(self):

        #with open("dataset/corpus_chatbot_apotek.json", "r") as f:
        data_file = open("dataset/corpus_chatbot_apotek.json").read()
        data = json.loads(data_file)

        #data = data_json["data"]
        
        words = []
        classes = []
        data_X = [] 
        data_y = []

        #pre_processing = Preprocessing()

        #for d in data:
        #    tag = d["Version ID"]
        #    classes.append(tag)
        #    for pattern in d["questions_pattern"]:
        #        w = pre_processing.tokenize_text(pattern)
        #        words.extend(w) 
        #        #xy.append((w, tag))
        #        data_X.append(pattern)
        #        data_y.append(d["Version ID"])

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                words.extend(tokens)
                data_X.append(pattern)
                data_y.append(intent["tag"])
    
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

        # create stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        words = [stemmer.stem(w) for w in words]
        #words = words[0]
        #words = list(dict.fromkeys(words))
        #classes = list(dict.fromkeys(classes))

        words = sorted(set(words))
        classes = sorted(set(classes))
        #words.sort()
        #classes.sort()

        training = []
        out_empty = [0] * len(classes)

        for idx, doc in enumerate(data_X):
            bow = []
            text = stemmer.stem(doc)
            for word in words:
                bow.append(1) if word in text else bow.append(0)
        
            output_row = list(out_empty)
            output_row[classes.index(data_y[idx])] = 1
    
            training.append([bow, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)

        train_X = np.array(list(training[:,0]))
        train_Y = np.array(list(training[:,1]))

        model = Sequential()
        model.add(Dense(128, input_shape=(len(train_X[0]),), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(train_Y[0]), activation="softmax"))

        adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

        print(model.summary())
        model.fit(x=train_X, y=train_Y, epochs=150, verbose=1)
        model.save('dataset/data_model')

        #return model, words, classes, data
        #return model


