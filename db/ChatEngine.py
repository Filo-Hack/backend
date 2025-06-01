from dbManager import ChromaDBManager
import os
from config import settings

import json
from typing import List,Dict,Any
from loguru import logger
class ChatEngine:
    def __init__(self, chroma_db: ChromaDBManager):
        self.chroma_db = chroma_db

    def generate_response(self, query: str) -> str:
        context = self.get_relevant_context(query)
        formatted = self.format_context(context)
        prompt = f"Контекст:\n{formatted}\n\nВопрос: {query}\nОтвет:"
        return prompt

    def save_record(self, data_json):
        try:
            data = data_json.json() if hasattr(data_json, "json") else data_json

            os.makedirs(settings.PARSED_JSON_PATH, exist_ok=True)
            file_path = os.path.join(settings.PARSED_JSON_PATH, f"{data['id']}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Получаем вектор эмбеддинга голоса
            voice_vector = data.get("voice_embedding")  # должен быть список чисел

            if voice_vector is None:
                logger.error("В данных отсутствует 'voice_embedding'")
                return

            # Поиск похожих по вектору
            results = ChatEngine.chroma_db.similarity_search_by_vector(
                embedding=voice_vector,
                k=5,
                filter={}  # например, можно добавить фильтр по другим метаданным
            )

            logger.success(f"Найдено {len(results)} похожих записей по голосу")
            return results

        except Exception as e:
            logger.error(f"Ошибка при сохранении и поиске по голосу: {e}")
            raise

    from typing import Optional

    def get_relevant_context(
            self,
            query: Optional[str] = None,
            voice_embedding: Optional[List[float]] = None,
            k: int = 7
    ) -> List[Dict[str, Any]]:
        try:
            results = []
            if voice_embedding is not None:
                results = self.chroma_db.similarity_search_by_vector(voice_embedding, k=k)
            elif query is not None:
                results = self.chroma_db.similarity_search(query, k=k)
            else:
                logger.warning("Не передан ни текст, ни голос для поиска.")
                return []

            return [{"text": doc.page_content, "metadata": doc.metadata} for doc in results]

        except Exception as e:
            logger.error(f"Ошибка при поиске контекста: {e}")
            return []

    def format_context(self, context: List[Dict[str, Any]]) -> str:
        formatted = []
        for item in context:
            meta = "\n".join(f"{k}: {v}" for k, v in item["metadata"].items())
            formatted.append(f"Текст: {item['text']}\nМетаданные:\n{meta}\n")
        return "\n---\n".join(formatted)
