import pandas as pd
import numpy as np
import requests
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer, LancasterStemmer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download("punkt")
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import spacy
from spacy.lang.id import Indonesian
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.tokenizer import Tokenizer, PhraseTokenizer # belum dicoba

### PREPROCESSING
class Preprocessing:
    def clean_text(self, texts):
        # text lower case
        text_clean = texts.lower()
        # get only alphabet text
        text_clean = re.sub("[^A-Za-z]+", " ", text_clean)
        
        return text_clean

    def tokenize_text(self, texts):
        all_sentences = nltk.sent_tokenize(texts)
        all_words = [nltk.word_tokenize(sent) for sent in all_sentences]
        return all_words

    def remove_wordstop(self, words):
        # stopword dari tala
        sw1_raw = pd.read_csv("https://raw.githubusercontent.com/masdevid/ID-Stopwords/master/id.stopwords.02.01.2016.txt", lineterminator="\n", names=["stopword"], header=None)
        sw1 = sw1_raw["stopword"].values.tolist()
        # print(len(sw1))

        # stopword dari sastrawi
        stop_factory = StopWordRemoverFactory()
        sw2 = stop_factory.get_stop_words()
        # print(len(sw2))

        # stopword dari spacy id
        sw3_raw = Indonesian()
        sw3_raw = sw3_raw.Defaults.stop_words
        sw3 = list(sw3_raw)
        # print(len(sw3))

        # stopword gabungan
        sw_raw = sw2+sw1+sw3
        # print(len(sw_raw))
        sw = list(dict.fromkeys(sw_raw))
        stopwords_list = sw

        for i in range(len(words)):
            words[i] = [w for w in words[i] if w not in stopwords_list]
        return words    

    def lemmatization_text(self, words):
        lemmatizer = Lemmatizer()
        lemma_text = lemmatizer.lemmatize(words)
        return lemma_text

    def stemming_text(self, words):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stem_text   = stemmer.stem(words)
        return stem_text

    def pre_process(self, docs):
        text_processed = str(docs)
        text_processed = self.clean_text(text_processed)
        text_processed = self.stemming_text(text_processed)
        text_processed = self.tokenize_text(text_processed)
        text_processed = self.remove_wordstop(text_processed)
        text_processed = text_processed[0]
        return text_processed