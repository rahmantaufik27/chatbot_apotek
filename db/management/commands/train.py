from bot import train_elsa
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    def handle(self, *args, **options):
        train_elsa()
