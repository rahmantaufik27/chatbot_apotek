import pandas as pd
import numpy as np 
import random 
import json
import nltk
from tensorflow import keras
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class Prediction: 
    
    # inisiasi stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # cleaning text ke tokens
    def clean_text(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [self.stemmer.stem(word) for word in tokens]
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
        model = keras.models.load_model('dataset/data_model')
        bow = self.bag_of_words(text, vocab)
        result = model.predict(np.array([bow]))[0]
        thresh = 0.235
        y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
        y_pred.sort(key=lambda x: x[1], reverse=True)
    
        return_list = []
        for r in y_pred:
            return_list.append(labels[r[0]])
    
        return return_list

    # respon jawaban yang akan ditampilkan
    def get_response(self, intents_list, intents_json):
        if len(intents_list) == 0:
            result = "Mohon maaf! pertanyaan yang anda masukan tidak ada jawabannya"
        else:
            tag = intents_list[0]
            list_of_intents = intents_json["intents"]
        
            for i in list_of_intents:
                if i["tag"] == tag:
                    result = random.choice(i["responses"])
                    break
        return result

    # jawaban berdasarkan inputan pertanyaan
    def get_result(self, message):
        words = []
        classes = []
        data_file = open("dataset/corpus_chatbot_apotek.json").read()
        data = json.loads(data_file)

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                words.extend(tokens)
    
            if intent["tag"] not in classes:
                classes.append(intent["tag"])
            
        words = [self.stemmer.stem(word.lower()) for word in words]
        words = sorted(set(words))
        classes = sorted(set(classes))
        
        intents = self.pred_class(message, words, classes)
        result = self.get_response(intents, data)
        #print(self.words)
        return result