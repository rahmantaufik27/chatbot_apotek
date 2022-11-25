from dataset import CLEAN_DATA

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    def handle(self, *args, **options):
        CLEAN_DATA.to_csv("dataset_examples.csv", index=False)
