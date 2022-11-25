from django.core.management.base import BaseCommand

from db.models import DatasetDb
import pandas as pd


class Command(BaseCommand):
    def handle(self, *args, **options):
        dataset: pd.DataFrame = pd.read_csv("dataset_examples.csv")
        dataset = dataset.rename(
            columns={
                "Pertanyaan": "pertanyaan",
                "Jawaban": "jawaban",
                "No": "uid",
            }
        )

        dataset = dataset.drop(columns="Unnamed: 0")
        dataset_objects = [
            DatasetDb(**values) for values in dataset.to_dict(orient="records")
        ]
        DatasetDb.objects.bulk_create(
            dataset_objects,
            ignore_conflicts=True
        )
