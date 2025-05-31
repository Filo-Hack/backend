import sys
from loguru import logger
from .config import CHROMADB_PERSIST_DIR, DEVICE

# Конфигурация логгера
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    level="INFO"
)

logger.info(f"Используем persist directory для ChromaDB: {CHROMADB_PERSIST_DIR}")
logger.info(f"Устройство для PyTorch: {DEVICE}")