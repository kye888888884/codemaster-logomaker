from django.apps import AppConfig
from .deepmodel import DeepModel

class MainConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "main"
    deepmodel = DeepModel()

