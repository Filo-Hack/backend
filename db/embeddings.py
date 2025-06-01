from langchain_huggingface import HuggingFaceEmbeddings
from config import settings
embeddings = HuggingFaceEmbeddings(
    model_name=settings.LM_MODEL_NAME,
    model_kwargs={"device": "cuda"}  # или "cuda" если у тебя есть GPU
)

