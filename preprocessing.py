import pandas as pd
import numpy as np
import requests
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer, LancasterStemmer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import spacy
from spacy.lang.id import Indonesian
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from nlp_id.lemmatizer import Lemmatizer
from nlp_id.tokenizer import Tokenizer, PhraseTokenizer # belum dicoba

nltk.download("punkt")
nltk.download('stopwords')

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
        # stopword dari tala, sastrawi dan spacy id
        sw1_raw = pd.read_csv("dataset/stopwords_combination.txt", lineterminator="\n", names=["stopword"], header=None)
        sw1 = sw1_raw["stopword"].values.tolist()
        # stopword apotek custom
        sw4_raw = pd.read_csv("dataset/stopwords_apotek_custom.txt", lineterminator="\n", names=["stopword"], header=None)
        sw4 = sw4_raw["stopword"].values.tolist()
        # stopword gabungan indonesia
        sw_raw = sw1+sw4
        # hapus duplikat values di list
        sw = list(dict.fromkeys(sw_raw))
        # stopword gabungan indonesia inggris
        stopwords_list = nltk.corpus.stopwords.words("english")
        stopwords_list.extend(sw)

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
        # text_processed = self.stemming_text(text_processed)
        text_processed = self.tokenize_text(text_processed)
        text_processed = self.remove_wordstop(text_processed)
        text_processed = text_processed[0]
        return text_processed