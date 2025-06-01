from dbManager import ChromaDBManager
from ChatEngine import ChatEngine
from config import settings
if __name__ == "__main__":

    db = ChromaDBManager()
    chat = ChatEngine(chroma_db=db)
    docs = ChromaDBManager.get(include=['embeddings'])
    print(docs["embeddings"][0])
    question = "Что произошло на конференции?"
    answer = chat.generate_response(question)
    print(answer)
