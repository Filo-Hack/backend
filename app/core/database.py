import os
import sys
from loguru import logger
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
from .config import CHROMADB_PERSIST_DIR

# Инициализация ChromaDB (PersistentClient)

try:
    client = chromadb.PersistentClient(
        path=CHROMADB_PERSIST_DIR,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    logger.info("ChromaDB: PersistentClient создан.")
except Exception as e:
    logger.error(f"Ошибка при создании PersistentClient: {e}")
    sys.exit(1)

try:
    collection = client.get_or_create_collection(name="user_profiles")
    logger.info("ChromaDB: коллекция 'user_profiles' готова.")
except Exception as e:
    logger.error(f"Ошибка при создании/получении коллекции: {e}")
    sys.exit(1)