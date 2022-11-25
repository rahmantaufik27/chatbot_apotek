import django
import os
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID
from enum import Enum

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from db.models import SessionsDb, BotSessionDb  # noqa


class SessionEnum(str, Enum):
    ENDED = "END"
    CONT = "CONTINUE"


class BaseSession(BaseModel):
    id: str | UUID = None
    created_at: datetime = None
    updated_at: datetime = None


class SessionCreate(BaseModel):
    name: str
    gender: str

    class Config:
        orm_mode = True

    @classmethod
    def get(cls, **kwargs):
        session: SessionsDb = SessionsDb.objects.get(**kwargs)
        return cls.from_orm(session)

    def create(self):
        session: SessionsDb = SessionsDb.objects.create(
            name=self.name, gender=self.gender
        )
        return session.pk


class Sessions(SessionCreate, BaseSession):
    pass


class BotSessionCreate(BaseModel):
    session_id: str
    question: str = "hi, elsa"
    session: SessionEnum = SessionEnum.CONT


class BotSession(BaseModel):
    id: int = None
    session_id: str | UUID = None
    question: str
    answer: str
    choices: list = Field(default_factory=list)

    class Config:
        orm_mode = True

    @classmethod
    def get(cls, **kwargs):
        session: BotSessionDb = BotSessionDb.objects.get(**kwargs)
        return cls.from_orm(session)

    def create(self, greetings: bool = False, finish: bool = False):
        botsession: BotSessionDb = BotSessionDb.objects.create(
            question=self.question,
            answer=self.answer,
            session_id=self.session_id,
            greetings=greetings,
            finish=finish,
        )
        return botsession.pk
