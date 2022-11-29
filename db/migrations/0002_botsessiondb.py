# Generated by Django 4.1.2 on 2022-10-12 03:12

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("db", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="BotSessionDb",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("question", models.TextField(blank=True, null=True)),
                ("answer", models.TextField(blank=True, null=True)),
                ("bot_say", models.TextField(blank=True, null=True)),
                ("greetings", models.BooleanField(default=False)),
                ("finish", models.BooleanField(default=False)),
                (
                    "session",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE, to="db.sessionsdb"
                    ),
                ),
            ],
        ),
    ]