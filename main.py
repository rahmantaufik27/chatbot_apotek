import pandas as pd
import numpy as np 
import json 
import nltk
from tensorflow import keras
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from preprocessing import Preprocessing
from postprocessing import Postprocessing
from modeling import Modeling_rule_based, Modeling_learning_based
from prediction import Prediction
#from generate_model import Generate_model 

# GENERATE CORPUS AND MODEL (ANSWERED - QUESTIONS PATTERN)

# ANSWER PREDICTION FROM QUESTION (BASED ON MODEL)
## inisiasi
model = keras.models.load_model('dataset/data_model')
data_file = open("dataset/corpus_chatbot_apotek.json").read()
data = json.loads(data_file)        
words = []
classes = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
    
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

factory = StemmerFactory()
stemmer = factory.create_stemmer()

words = [stemmer.stem(w) for w in words]
words = list(dict.fromkeys(words))
classes = list(dict.fromkeys(classes))

predition = Prediction()
message = "bagaimana menyimpan erha4"
intents = predition.pred_class(message, words, classes)

result = predition.get_response(intents, data)
print(result)
