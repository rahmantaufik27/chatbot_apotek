import numpy as np
import random
from preprocessing import Preprocessing
from postprocessing import Postprocessing
from modeling import Modeling_rule_based, Modeling_learning_based
from tensorflow import keras
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

class Prediction:
    def clean_text(self, text):
        
        factory = StemmerFactory()
        stemmer = factory.create_stemmer() 

        tokens = nltk.word_tokenize(text)
        tokens = [stemmer.stem(w) for w in tokens]
        return tokens

    def bag_of_words(self, text, vocab):
        tokens = self.clean_text(text)
        bow = [0] * len(vocab)
        for w in tokens:
            for idx, word in enumerate(vocab):
                if word == w:
                    bow[idx] = 1
        return np.array(bow)

    def pred_class(self, text, vocab, labels):
        bow = self.bag_of_words(text, vocab)
        #data_model = Modeling_learning_based() 
        #model = data_model.training_model()
        model = keras.models.load_model('dataset/data_model')
        result = model.predict(np.array([bow]))[0]
        thresh = 0.5
        y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
        y_pred.sort(key=lambda x: x[1], reverse=True)
    
        return_list = []
        for r in y_pred:
            return_list.append(labels[r[0]])
    
        return return_list

    def get_response(self, intents_list, intents_json):
        if len(intents_list) == 0:
            result = "Maaf kata kunci yang anda masukan tidak ada jawabannya"
        else:
            tag = intents_list[0]
            list_of_intents = intents_json["intents"]
        
            for i in list_of_intents:
                if i["tag"] == tag:
                    result = random.choice(i["responses"])
                    break
        return result