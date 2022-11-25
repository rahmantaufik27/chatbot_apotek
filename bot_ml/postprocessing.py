# flake8: noqa
import pandas as pd
import numpy as np
import json
import requests
import sys
import re


class Postprocessing:
    def generate_corpus(self, df):
        try:
            df.to_json("dataset/corpus_chatbot_apotek_extended.json", orient="table")
        except Exception as e:
            print("Output error")
            exit
