from django.db import models
import uuid


class DatasetDb(models.Model):
    uid = models.CharField(max_length=255, unique=True)
    pertanyaan = models.TextField(blank=True, null=True)
    jawaban = models.TextField(blank=True, null=True)
    spec = models.CharField(max_length=255)
    product_category = models.CharField(max_length=255)
    question_category = models.CharField(max_length=255)

    @property
    def spec_bot(self):
        return self.spec.replace(" ", "_")


class SessionsDb(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    gender = models.CharField(max_length=20)
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name


class BotSessionDb(models.Model):
    session: SessionsDb = models.ForeignKey(SessionsDb, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    question = models.TextField(blank=True, null=True)
    prev_question = models.ForeignKey(
        DatasetDb, on_delete=models.CASCADE, null=True, blank=True
    )
    answer = models.TextField(blank=True, null=True)
    state = models.CharField(max_length=255)
    greetings = models.BooleanField(default=False)
    finish = models.BooleanField(default=False)
