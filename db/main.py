from dbManager import ChromaDBManager
from ChatEngine import ChatEngine
from config import settings
if __name__ == "__main__":
    db = ChromaDBManager()
    chat = ChatEngine(chroma_db=db)

    db.add_documents_from_json(settings.PARSED_JSON_PATH)

    question = "Что произошло на конференции?"
    answer = chat.generate_response(question)
    print(answer)
