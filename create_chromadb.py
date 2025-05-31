import time
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from loguru import logger
from langchain_chroma import Chroma

CHROMA_PATH = "./shop_chroma_db"
COLLECTION_NAME = "shop_data"

SHOP_DATA = [
    {
        "text": "хочу заказать доставку",
        "metadata": {
            "id": "u001",
            "type": "voice_command",
            "speaker": "Никита",
            "voice_embedding": [0.13, 0.41, ...],  # список float — эмбеддинг тембра
            "timestamp": "2025-05-31T10:45:00",
        },
    },
    {
        "text": "отменить заказ",
        "metadata": {
            "id": "u002",
            "type": "voice_command",
            "speaker": "Ольга",
            "voice_embedding": [0.22, 0.44, ...],
            "timestamp": "2025-05-31T11:12:00",
        },
    },
]



def generate_chroma_db():
    try:
        start_time = time.time()

        logger.info("Загрузка модели эмбеддингов...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Модель загружена за {time.time() - start_time:.2f} сек")

        logger.info("Создание Chroma DB...")
        chroma_db = Chroma.from_texts(
            texts=[item["text"] for item in SHOP_DATA],
            embedding=embeddings,
            ids=[str(item["metadata"]["id"]) for item in SHOP_DATA],
            metadatas=[item["metadata"] for item in SHOP_DATA],
            persist_directory=CHROMA_PATH,
            collection_name=COLLECTION_NAME,
        )
        logger.info(f"Chroma DB создана за {time.time() - start_time:.2f} сек")

        return chroma_db
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise
