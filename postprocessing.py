import pandas as pd 
import numpy as np 
import json 
import requests
import sys

# df = pd.read_excel("dataset/Chatbot_node_wording.xlsx")

class Postprocessing:
    def post_process(self, df):
        try:
            df.to_json("dataset/chatbot_output_2.json", orient="table")
        except Exception as e:
            print("Output error")
            exit

    # result = df.to_json(orient="table")
    # parsed = json.loads(result)
    # dumped = json.dumps(parsed, indent=4)
    # print(dumped)
    # df_res = pd.read_json("dataset/chatbot_output.json", orient ="table")
    # print(df_res)