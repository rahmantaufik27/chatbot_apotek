import pandas as pd
import numpy as np 

from preprocessing import Preprocessing
from postprocessing import Postprocessing

# 1. LOAD DATASET
df = pd.read_excel("dataset/Chatbot_node_wording.xlsx")

# 2. PREPROCESSING DATASET
## get cleaned tokenized word for learning model 
pre_processing = Preprocessing()
df["answer_tokenized"] = df["Answer 1"].apply(lambda x: pre_processing.pre_process(x))
# print(df.info())
# print(df.head())

# 3. MODELING DATASET
## training (create corpus for questions), and testing

# 4. POSTPROCESSING DATASET
## transform into json file
df_res = df[["Content ID", "Answer 1", "Next Event", "Next Data", "Version ID", "Condition 1 Operator", "Condition 1 Property", "Condition 1 Value", "answer_tokenized"]]
post_processing = Postprocessing()
post_processing.post_process(df_res)