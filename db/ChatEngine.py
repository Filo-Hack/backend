class ChatEngine:
    def __init__(self, chroma_db: ChromaDBManager):
        self.chroma_db = chroma_db
        self.llm = ...  # сюда ваш LLM

    def generate_response(self, query: str) -> str:
        context = self.get_relevant_context(query)
        formatted = self.format_context(context)
        prompt = f"Контекст:\n{formatted}\n\nВопрос: {query}\nОтвет:"
        return self.llm.generate(prompt)

    def save_record(self, data_json):
        """
        Принимает JSON-ответ от сервера и сохраняет каждый документ в отдельный .json файл.
        """
        try:
            if hasattr(data_json, "json"):
                data = data_json.json()
            else:
                data = data_json

            os.makedirs(settings.PARSED_JSON_PATH, exist_ok=True)

            for i, item in enumerate(data):
                file_path = os.path.join(settings.PARSED_JSON_PATH, f"doc_{i}.json")
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)

            logger.success(f"Сохранено {len(data)} JSON-документов")
        except Exception as e:
            logger.error(f"Ошибка при сохранении JSON: {e}")
            raise

    def get_relevant_context(self, query: str, k: int = 7) -> List[Dict[str, Any]]:
        try:
            results = self.chroma_db.similarity_search(query, k=k)
            return [
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Ошибка при поиске контекста: {e}")
            return []

    def format_context(self, context: List[Dict[str, Any]]) -> str:
        formatted = []
        for item in context:
            meta = "\n".join(f"{k}: {v}" for k, v in item["metadata"].items())
            formatted.append(f"Текст: {item['text']}\nМетаданные:\n{meta}\n")
        return "\n---\n".join(formatted)
