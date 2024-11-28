# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load variables
API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
