import pandas as pd 
import numpy as np 
import json 
import requests
import sys
import re

class Postprocessing:
    #def generate_corpus(self, df):
    #    #try:
    #    #    df.to_json("dataset/chatbot_question_answer_pattern.json", orient="table")
    #    #except Exception as e:
    #    #    print("Output error")
    #    #    exit

    #    def unit_transform(str):
    #        replacements = {
    #                '[0-9]+,' : '.',
    #            }

    #        for key, value in replacements.items():
    #            str = re.sub(key, value, str)
    #        return (str)
        
    #    contexts = df['questions_pattern']
    #    tags = df['Version ID']
    #    responses = df['Answer 1']
    #    result = "{\"intents\": ["

    #    for context, tag, response in zip(contexts, tags, responses):
    #        result = result + "{\"context\": \"\",\"patterns\": " + str(context) + ",\"responses\": "+ "[\""+ str(response) +"\"]" +",\"tag\": \""+ str(tag) +"\"},"

    #    result = r"{}".format(result)
    #    result = result.replace("\'", "\"")
    #    result = result.replace("\'", "\"")

    #    result = unit_transform(result) + "]}"

    #    f = open("dataset/corpus_chatbot_apotek.json", "w")
    #    f.write(str(result.encode('ascii', 'replace')))
    #    f.close()

    #    #return result
    def unit_transform(self, str):
        replacements = {
                '[0-9]+,' : '.',
            }

        for key, value in replacements.items():
            str = re.sub(key, value, str)
        return (str)

    def generate_json(self, df):
        contexts = df['questions_pattern']
        tags = df['Version ID']
        responses = df['Answer 1']
        result = "{\"intents\": ["

        for context, tag, response in zip(contexts, tags, responses):
            result = result + "{\"context\": \"\",\"patterns\": " + str(context) + ",\"responses\": "+ "[\""+ str(response) +"\"]" +",\"tag\": \""+ str(tag) +"\"},"

        result = r"{}".format(result)
        result = result.replace("\'", "\"")
        result = result.replace("\'", "\"")

        return self.unit_transform(result) + "]}"

    def generate_corpus(self, df):
        f = open("data.json", "w")
        f.write(str(self.generate_json(df).encode('ascii', 'replace')))
        f.close()