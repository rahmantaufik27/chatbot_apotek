import argparse

from preprocessing import Preprocessing
from postprocessing import Postprocessing
from modeling import Modeling_rule_based, Modeling_learning_based, Generate_model
from prediction import Prediction

parser = argparse.ArgumentParser()
parser.add_argument('--train_model', action='store_true', help='generate corpus and data model')
parser.add_argument('--testing_model', action='store_true', help='testing model for question-answers based on data model')
args = parser.parse_args()

if __name__ == '__main__':

    if args.train_model:
        # GENERATE CORPUS AND MODEL (ANSWERED - QUESTIONS PATTERN)
        model = Generate_model()
        model.generate()
    
    elif args.testing_model:
        # ANSWER PREDICTION FROM QUESTION (BASED ON MODEL)
        predict_answer = Prediction()
        #question = "corrective cream 5 10 g"
        question = "kasih tau cara pake acl dong"
        res = predict_answer.get_result(question)
        print(res)