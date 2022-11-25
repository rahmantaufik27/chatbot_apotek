from chatterbot.storage import SQLStorageAdapter


class Tagger:
    ISO_639_1 = "en_core_web_sm"


class SqliteAdapter(SQLStorageAdapter):
    def __init__(self, *args, **kwargs):
        kwargs.update({"tagger_language": Tagger})
        super().__init__(**kwargs)
