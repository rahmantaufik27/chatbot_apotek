import pandas as pd
import numpy as np 

from preprocessing import Preprocessing
from postprocessing import Postprocessing
from modeling import Modeling_rule_based, Modeling_learning_based

class Modeling_learning_based:
    def wawa(self):
        printw = "haha"
        return printw 

class Call_learning_based:
    def wuwu(self):
        haha = Modeling_learning_based()
        print(haha.wawa())


#class Generate_model_learning:
#    # GENERATE CORPUS (ANSWERED - QUESTIONS PATTERN)
#    # 1. LOAD DATASET
#    df_raw = pd.read_excel("dataset/Chatbot_node_wording.xlsx")
#    ## filter data (ignore null answers, and get only product_info and treatment_info)
#    df_raw = df_raw.loc[df_raw["Content ID"].isin(["product_info", "treatment_info"])]
#    df_raw = df_raw.loc[~df_raw[["Answer 1"]].isna().any(axis=1)]
#    print(len(df_raw))

#    ## Process per Batch
#    batch_size = 100
#    pre_processing = Preprocessing()
#    rule_based = Modeling_rule_based()
#    df = pd.DataFrame()

#    for batch_number, batch_df in df_raw.groupby(np.arange(len(df_raw)) // batch_size):
#        # 2. PREPROCESSING DATASET
#        ## get cleaned tokenized word for learning model 
#        batch_df["answer_tokenized"] = batch_df["Answer 1"].apply(lambda x: pre_processing.pre_process(x))

#        # 3. MODELING DATASET
#        ## training (create corpus for questions), and testing
#        batch_df["product_name"] = batch_df["Answer 1"].apply(lambda x: rule_based.get_product(x))
#        batch_df.loc[batch_df["product_name"] == "", "product_name"] = np.nan
#        batch_df.loc[batch_df["product_name"] == " ", "product_name"] = np.nan
#        batch_df["product_name"].fillna(batch_df["Version ID"], inplace=True)
#        batch_df["questions_candidate"] = batch_df.apply(lambda x: rule_based.create_questions(x["Content ID"], x["product_name"]), axis=1)
#        batch_df["questions_pattern"] = batch_df.apply(lambda x: rule_based.create_pattern_questions(x["answer_tokenized"], x["product_name"], x["questions_candidate"]), axis=1)
#        df = pd.concat([df, batch_df], ignore_index=True)
#    print(len(df))

#    # 4. GENERATE CORPUS FROM DATASET (QUESTIONS CANDIDATE)
#    ## transform into json file
#    df_res = df[["Content ID", "Answer 1", "Version ID", "product_name", "questions_pattern"]]
#    #post_processing = Postprocessing()
#    #post_processing.generate_corpus(df_res)

#    # 5. TRAINING TO GET DATA MODEL (QUESTION ANSWER RELATION)
#    modeling_ml = Modeling_learning_based() 
#    modeling_ml.training_model()
