from dataset import get_dataset
from chatterbot import ChatBot

from chatterbot.trainers import ListTrainer

CLEAN_DATA = get_dataset()
CLEAN_DATA["spec"] = (
    CLEAN_DATA["product_category"].astype(str)
    + " "
    + CLEAN_DATA["question_category"].astype(str)
)
# CLEAN_DATA.to_csv("dataset_examples.csv", index=False)
greet_words = ["hi, elsa", "hallo", "hi bot"]
greeting_answer = CLEAN_DATA.loc[CLEAN_DATA["No"] == "A.1"]["Jawaban"][0]
greeting_answer_2 = CLEAN_DATA.loc[CLEAN_DATA["No"] == "B.1"].reset_index()["Jawaban"][
    0
]
greeting_p = CLEAN_DATA.loc[CLEAN_DATA["No"] == "B.1"].reset_index()["Pertanyaan"][0]
content = ["program", "produk", "greeting"]
bot_content = CLEAN_DATA.query("question_category not in @content")


def print_choices(choices: list):
    for idx, choice in enumerate(choices):
        choice = choice.replace("_", " ")
        print(f"{idx}. : {choice}")


def is_valid_choice(input):
    try:
        int(input)
        return True
    except ValueError:
        return False


def train_solution(trainer):
    training_method = ["program", "produk"]
    bot_solution = CLEAN_DATA.query("question_category == 'solution'")
    for index, solution in bot_solution.iterrows():
        for method in training_method:
            product_category = solution["product_category"]
            data = CLEAN_DATA.query(
                "product_category == @product_category and question_category == @method"
            ).reset_index()
            if not data.empty:
                spec_id = data["spec"][0].replace(" ", "_")
                trainer.train([spec_id, data["Jawaban"][0]])


def train_give_solution_question(trainer):
    bot_solution = CLEAN_DATA.query("question_category == 'solution'")
    for index, solution in bot_solution.iterrows():
        trainer.train([solution["spec"].replace(" ", "_"), solution["Pertanyaan"]])


def train_non_solution_question(trainer):
    bot_solution = CLEAN_DATA.query(
        "question_category in ('content', 'promotion' ,'innovation','find us' , 'closing')"
    )
    for index, solution in bot_solution.iterrows():
        trainer.train([solution["spec"].replace(" ", "_"), solution["Pertanyaan"]])


ELSA = ChatBot(
    "ELSA",
    storage_adapter="adapter.Sqlite.SqliteAdapter",
    logic_adapters=[
        "chatterbot.logic.BestMatch",
    ],
)


def train_elsa():
    trainer = ListTrainer(ELSA)
    for word in greet_words:
        trainer.train([word, greeting_answer])
    train_solution(trainer)
    train_give_solution_question(trainer)
    train_non_solution_question(trainer)


# The following loop will execute each time the user enters input
def run():
    choices = []
    greet = False
    choices = []
    prev_q = None
    choice_query = None
    while True:
        try:
            if greet and prev_q is None:
                print_choices(bot_content["spec"].to_list())
                choices = bot_content.to_dict(orient="records")

            if greet and choice_query:
                print_choices(choice_query)
                choices = choice_query
                prev_q = None

            user_input = input("you: ")
            if prev_q is not None:
                # res = chatbot.get_response(prev_q)
                res = ELSA.get_response(user_input)
                print("ELSA : -> ", res)
                prev_q = None
                continue

            if is_valid_choice(user_input):
                q = choices[int(user_input)]
                print("ELSA : -> ", q["Pertanyaan"])
                prev_q = q["Pertanyaan"]
                p_category = q["product_category"]
                choice_query = CLEAN_DATA.query(
                    "product_category == @p_category and question_category in (@content)"
                )["spec"].to_list()
                continue

            if user_input in greet_words:
                bot_response = ELSA.get_response(user_input)
                print("ELSA : -> ", bot_response)
                greet = True
        # Press ctrl-c or ctrl-d on the keyboard to exit
        except (KeyboardInterrupt, EOFError, SystemExit):
            break
