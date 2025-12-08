from dotenv import load_dotenv
import os

load_dotenv()

class Configs:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
    VECTOR_DIR = os.getenv("VECTOR_INDEX_DIR", "data/vectorstore")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    REDIS_URL = os.getenv("REDIS_URL")

configs = Configs()
