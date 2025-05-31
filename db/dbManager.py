from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Стало:
from langchain_community.vectorstores import Chroma


from langchain.schema.document import Document
from typing import List, Dict, Any
import os
import json
from config import settings
from embeddings import embeddings  # ваш эмбеддинг, например OpenAIEmbeddings


def split_text_into_chunks(text: str, metadata: Dict[str, Any]) -> List[Document]:
    """Разделение текста на чанки с сохранением метаданных."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.MAX_CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents(texts=[text], metadatas=[metadata])
    return chunks


class ChromaDBManager:
    def __init__(self):
        self.chroma = Chroma(
            persist_directory=settings.AMVERA_CHROMA_PATH,
            embedding_function=embeddings,
            collection_name=settings.AMVERA_COLLECTION_NAME
        )
        logger.success("Chroma DB инициализирована")

    def add_documents_from_json(self, json_dir: str):
        """Загрузка всех .json файлов и добавление в базу с разбиением на чанки."""
        all_chunks = []

        for filename in os.listdir(json_dir):
            if not filename.endswith(".json"):
                continue

            path = os.path.join(json_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)

                text = doc_data.get("text", "")
                metadata = {k: v for k, v in doc_data.items() if k != "text"}
                chunks = split_text_into_chunks(text, metadata)
                all_chunks.extend(chunks)

                logger.info(f"{filename} → {len(chunks)} чанков добавлено")
            except Exception as e:
                logger.error(f"Ошибка при обработке {filename}: {e}")

        if all_chunks:
            self.chroma.add_documents(all_chunks)
            self.chroma.persist()
            logger.success(f"Всего добавлено {len(all_chunks)} чанков")

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        return self.chroma.similarity_search(query, k=k)
