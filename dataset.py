import pandas as pd

pd.set_option("mode.chained_assignment", None)
df: pd.DataFrame = pd.read_excel("dataset/worksheet.xlsx", skiprows=10)
df = df[
    [
        "No",
        "Kategori pertanyaan",
        "Sub Kategori Pertanyaan 1",
        "Sub Kategori Pertanyaan 2",
        "Pertanyaan",
        "Jawaban",
    ]
]

CATEGORIES = [
    "anti aging",
    "brightening",
    "acne care & cure",
    "hair",
    "dermatology",
    "men",
    "make over",
    "children",
    "skin",
]
CATEGORY_TYPES = ["solution", "product", "program"]
Q_CAT = [
    "greeting",
    "directory",
    "content",
    "promotion",
    "innovation",
    "find us",
    "closing",
]


def get_solution_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[df["Sub Kategori Pertanyaan 1"] == "solution".upper()]
    return df


def filter_solution(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[df["Sub Kategori Pertanyaan 1"] != "solution".upper()]
    return df


def put_service_categories(sub: str):
    sub = sub.lower()
    for category in CATEGORIES:
        if sub.find(category) > -1:
            replaced = sub.replace(category, "").strip()
            break
    replaced_list = replaced.split(" ")
    for type in CATEGORY_TYPES:
        if type in replaced_list:
            idx = replaced_list.index(type)
            result = replaced_list[idx]
            return result
    return CATEGORY_TYPES[0]


def put_categories(sub: str, listconstants: list):
    sub = sub.lower()
    for category in listconstants:
        if sub.find(category) > -1:
            return category


def clean_column(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[
            "Kategori pertanyaan",
            "Sub Kategori Pertanyaan 2",
            "Sub Kategori Pertanyaan 1",
        ]
    )


def get_cleaned_data() -> pd.DataFrame:
    non_sol = filter_solution(df)
    non_sol["product_category"] = "bot"
    non_sol["question_category"] = non_sol["Sub Kategori Pertanyaan 1"].apply(
        put_categories, listconstants=Q_CAT
    )
    solution = get_solution_df(df)
    solution = solution.dropna(subset="Sub Kategori Pertanyaan 2")

    solution["question_category"] = solution["Sub Kategori Pertanyaan 2"].apply(
        put_service_categories
    )
    solution["product_category"] = solution["Sub Kategori Pertanyaan 2"].apply(
        put_categories, listconstants=CATEGORIES
    )
    solution = clean_column(solution)
    non_sol = clean_column(non_sol)
    solution["Pertanyaan"] = solution["Pertanyaan"].fillna(method="ffill")
    solution["Jawaban"] = solution["Jawaban"].fillna(method="bfill")
    CLEAN_DATA = pd.concat([non_sol, solution])
    CLEAN_DATA["Jawaban"] = CLEAN_DATA["Jawaban"].str.replace("&#10;", " ")
    return CLEAN_DATA


def get_dataset() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv("dataset/dataset_examples.csv")
    return df
