
# Imports
from dotenv import load_dotenv
import os

from mistralai.client import MistralClient
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings



# Load environment variables from .env file
load_dotenv()
# Get the API key from environment variables
API_KEY = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=API_KEY)

