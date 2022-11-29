import argparse

from botmain.modeling import Generate_model, Generate_corpus, Modeling_word2vec, Modeling_chatterbot
from botmain.prediction import Prediction, Prediction_manual

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--generate_corpus", action="store_true", help="generate corpus"
)
parser.add_argument(
    "-tr",
    "--train_model",
    action="store_true",
    help="training model and generate to data model",
)
parser.add_argument(
    "-tt",
    "--testing_model",
    action="store_true",
    help="testing model for question-answers based on data model",
)
parser.add_argument(
    "-m",
    "--predict_manual",
    action="store_true",
    help="question-answer manual based on words frequency",
)
parser.add_argument(
    "-sw",
    "--similarity_word",
    action="store_true",
    help="get similarity word",
)
args = parser.parse_args()

if __name__ == "__main__":

    # corpus = Generate_corpus()
    # corpus.generate()

    # bot = Modeling_chatterbot()
    # bot.chat_bot()

    if args.generate_corpus:
        # GENERATE CORPUS (ANSWERED - QUESTIONS PATTERN)
        corpus = Generate_corpus()
        corpus.generate()

    elif args.train_model:
        # GENERATE DATA MODEL (BASED ON TRAINING CORPUS DATA)
        model = Generate_model()
        model.generate()

    elif args.testing_model:
        # PREDICT ANSWER AUTOMATICALLY FROM QUESTION (BASED ON DATA MODEL)
        predict_answer = Prediction()
        # baru bisa menjawab pertanyaan dengan nama produk nya spesifik
        # question = "informasi produk acne guard cream"
        # question = "kasih tau cara pakai anti comedonal lotion"
        # question = "produk moisturizer malam untuk kulit sensitif"
        # question = "treatment blue light therapy"
        # question = "terapi injeksi untuk mengurangi peradangan"
        question = "cara pakai produk erha 7"
        print(str(question))
        res, con = predict_answer.get_result(question)
        print(res)

        # jika tidak ada jawabannya lempar ke pertanyaan manual
        if con == 0:
            question = "cara pakai produk erha 7"
            print(str(question))
            predict_manual = Prediction_manual()
            res = predict_manual.get_answer(question)
            print(res)    

    elif args.predict_manual:
        # PREDICT ANSWER MANUALLY FROM QUESTION (BASED ON RULES)
        ## selama kata di pertanyaan sama dengan nama produknya, jawaban akan muncul tepat
        question = "cara pakai produk erha 7"
        # question = "harga dari erha truwhite ultimate radiance"
        # question = "greeting"
        print(str(question))
        predict_manual = Prediction_manual()
        res = predict_manual.get_answer(question)
        print(res)

    elif args.similarity_word:
        w2v = Modeling_word2vec()
        
        # training data
        # w2v.convert_data()
        # w2v.training_model()
        
        # testing data
        word = "bandung"
        sim_word = w2v.testing_model(word)
        print(sim_word)