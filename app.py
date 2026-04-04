"""
Application entry point.

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
from src.api.main import create_app

app = create_app()
