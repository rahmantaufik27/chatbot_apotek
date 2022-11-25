from fastapi import FastAPI, HTTPException
from schema import (
    Sessions,
    SessionsDb,
    SessionCreate,
    BotSessionCreate,
    BotSessionDb,
    BotSession,
    SessionEnum,
)
import uvicorn
from pipe import BotPipes

app = FastAPI(docs_url="/", redoc_url="/redoc")


@app.get("/session/{uid}", response_model=Sessions)
async def get_people(uid: str):
    p = await SessionsDb.objects.aget(pk=uid)
    people = Sessions.from_orm(p)
    return people


@app.post("/session", response_model=Sessions)
def create_session(s: SessionCreate):
    sid = s.create()
    session: Sessions = Sessions.get(pk=str(sid))
    return session


@app.post("/bot")
def bot_session(bs: BotSessionCreate):
    try:
        session = SessionsDb.objects.get(pk=bs.session_id)
    except (SessionsDb.DoesNotExist, Exception):
        raise HTTPException(status_code=404, detail="session not found")
    pipes = BotPipes(session, bs)
    result: BotPipes = pipes.validate().run()
    session_bot = BotSession.from_orm(result.bot_db)
    session_bot.choices = result.choices
    return session_bot


if __name__ == "__main__":
    uvicorn.run("serve:app", reload=True, host="0.0.0.0", port=8000)
    # http://localhost:8000/
