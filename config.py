import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
EMBEDDING_GOOGLE_API_KEY=os.getenv("EMBEDDING_GOOGLE_API_KEY")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

VIDEO_PATH = "test_video.mp4"
TEMP_AUDIO_DIR = "temp_audio"
SNAPSHOTS_DIR = "snapshots"
CLIPS_DIR = "clips"
REPORTS_DIR = "reports"
CHROMADB_DIR = "./chroma_db"

MCP_SERVER_HOST = "0.0.0.0"
MCP_SERVER_PORT = 8080
MCP_SERVER_URL = f"http://127.0.0.1:{MCP_SERVER_PORT}"