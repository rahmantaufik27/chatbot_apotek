# Generated by Django 4.1.2 on 2022-10-17 05:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("db", "0002_botsessiondb"),
    ]

    operations = [
        migrations.CreateModel(
            name="DatasetDb",
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
                ("uid", models.CharField(max_length=255, unique=True)),
                ("pertanyaan", models.TextField(blank=True, null=True)),
                ("jawaban", models.TextField(blank=True, null=True)),
                ("spec", models.CharField(max_length=255)),
                ("product_category", models.CharField(max_length=255)),
                ("question_category", models.CharField(max_length=255)),
            ],
        ),
        migrations.RenameField(
            model_name="botsessiondb",
            old_name="bot_say",
            new_name="prev_question",
        ),
        migrations.AddField(
            model_name="botsessiondb",
            name="state",
            field=models.CharField(default=1, max_length=255),
            preserve_default=False,
        ),
    ]
