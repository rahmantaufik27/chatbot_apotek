from db.models import BotSessionDb, SessionsDb, DatasetDb
from schema import BotSessionCreate
from bot import ELSA
from enum import Enum
from django.db.models import QuerySet
from chatterbot import ChatBot


class StateEnum(str, Enum):
    BEGIN = "begin"
    MAIN = "main"
    # in choices produk / program
    SERVICE = "service"
    # yes / no
    CHOICES = "choices"
    # ended
    ENDED = "ended"


class BotPipes:
    bot_db: BotSessionDb
    bot_sessions: QuerySet[BotSessionDb]
    answer: str = ""
    finish: bool = False
    greeting: bool = False
    choices: list = []
    payload: BotSessionCreate
    prev_question: DatasetDb = None
    session: SessionsDb
    state: str
    elsa: ChatBot = ELSA
    choice_select = ["AA", "TT"]

    def __init__(self, session: SessionsDb, payload: BotSessionCreate):
        self.session = session
        self.payload = payload

    def create_bot_session(self):
        self.bot_db = BotSessionDb.objects.create(
            session=self.session,
            question=self.payload.question,
            greetings=self.greeting,
            state=self.state,
            prev_question=self.prev_question,
            finish=self.finish,
            answer=self.answer,
        )

    def put_choices(self):

        self.choices = [
            {"name": "iya", "value": "AA"},
            {"name": "tidak", "value": "TT"},
        ]

        match self.state:
            case StateEnum.BEGIN | StateEnum.MAIN:
                self.choices = [
                    {"name": values.spec, "value": values.spec_bot}
                    for values in DatasetDb.objects.exclude(
                        question_category__in=[
                            "greeting",
                            "produk",
                            "program",
                        ]
                    )
                ]
            case StateEnum.SERVICE:
                p_cat = ""
                if self.prev_question:
                    p_cat = self.prev_question.product_category
                self.choices = [
                    {"name": values.question_category, "value": values.spec_bot}
                    for values in DatasetDb.objects.filter(
                        product_category=p_cat
                    ).exclude(question_category="solution")
                ]

    def validate(self):
        ended_bot_sessions = BotSessionDb.objects.filter(
            session_id=self.session.pk, finish=True
        )
        self.bot_sessions = BotSessionDb.objects.filter(session_id=self.session.pk)
        if ended_bot_sessions.exists():
            raise ValueError("session is finished")
        if not self.bot_sessions.exists():
            if self.payload.question != "hi, elsa":
                raise ValueError("session must start with greetings")
            self.greeting = True
            self.state = StateEnum.BEGIN
            self.create_bot_session()
        return self

    def get_latest_session(self) -> BotSessionDb:
        try:
            return self.bot_sessions.latest("created_at")
        except BotSessionDb.DoesNotExist:
            self.state = StateEnum.ENDED
            self.finish = True
            self.create_bot_session()
            raise ValueError("Latest Bot session not found")

    def run(self):
        latest_session: BotSessionDb = self.get_latest_session()

        match latest_session.state:
            case StateEnum.BEGIN:
                dataset = DatasetDb.objects.get(uid="A.1")
                self.answer = dataset.jawaban
                self.state = StateEnum.MAIN
                self.prev_question = dataset
            case StateEnum.MAIN:
                self.state = StateEnum.SERVICE
                self.answer = str(self.elsa.get_response(self.payload.question))
                try:
                    dataset = DatasetDb.objects.get(
                        spec=self.payload.question.replace("_", " ")
                    )
                except DatasetDb.DoesNotExist:
                    dataset = None
                self.prev_question = dataset

                if dataset.question_category != "solution":
                    self.state = StateEnum.CHOICES
            case StateEnum.SERVICE | StateEnum.CHOICES:
                if self.payload.question in self.choice_select:
                    dataset = DatasetDb.objects.get(
                        spec=latest_session.prev_question.spec
                    )
                    if self.payload.question == "TT":
                        self.answer = "Adakah yang bisa ELSA bantu lagi?"
                    if self.payload.question == "AA":
                        self.answer = dataset.jawaban
                else:
                    self.answer = str(self.elsa.get_response(self.payload.question))
                self.state = StateEnum.MAIN
        self.put_choices()
        self.create_bot_session()
        return self
