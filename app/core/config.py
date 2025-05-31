import os

# ─────────────────────────────────────────────────────────────────────────────
# Настройки и переменные окружения
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.getcwd())

CHROMADB_PERSIST_DIR = os.getenv(
    "CHROMADB_PERSIST_DIR",
    os.path.join(BASE_DIR, "chroma_data")
)
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "C:/проект/backend/models/vosk")
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
SUM_MODEL_NAME = os.getenv("SUM_MODEL_NAME", "google/flan-t5-small")
REC_MODEL_NAME = os.getenv("REC_MODEL_NAME", "google/flan-t5-small")
# TTS_MODEL_NAME = os.getenv("TTS_MODEL_NAME", "tts_models/ru/mai/tts")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"