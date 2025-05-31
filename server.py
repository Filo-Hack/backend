# server.py

import os
import asyncio
import base64
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger

import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from TTS.api import TTS

# ─────────────────────────────────────────────────────────────────────────────
# 1. НАСТРОЙКИ И ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# ─────────────────────────────────────────────────────────────────────────────

# Путь к папке, куда ChromaDB будет сохранять свои данные
CHROMADB_PERSIST = os.getenv(
    "CHROMADB_PERSIST_DIR",
    os.path.join(os.getcwd(), "chroma_data")
)

# Имена моделей
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUM_MODEL_NAME = "google/flan-t5-small"
REC_MODEL_NAME = "google/flan-t5-small"
TTS_MODEL_NAME = "tts_models/ru/ru-Rus/VITS"

# Устройство для PyTorch ("cuda" если есть GPU, иначе "cpu")
DEVICE = "cuda" if torch_available := (os.getenv("CUDA_AVAILABLE", "1") == "1") else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# 2. ИНИЦИАЛИЗАЦИЯ ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

# Убедимся, что директория существует
os.makedirs(CHROMADB_PERSIST, exist_ok=True)
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")

logger.info(f"Используем persist directory для ChromaDB: {CHROMADB_PERSIST}")

# Создаём клиент ChromaDB
client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMADB_PERSIST
    )
)

# Получаем или создаём коллекцию "user_profiles"
try:
    collection = client.get_collection(name="user_profiles")
    logger.info("ChromaDB: коллекция 'user_profiles' найдена.")
except ValueError:
    collection = client.create_collection(name="user_profiles")
    logger.info("ChromaDB: коллекция 'user_profiles' создана.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ЗАГРУЗКА МОДЕЛЕЙ (эмбеддинги, суммаризация, генерация, TTS)
# ─────────────────────────────────────────────────────────────────────────────

logger.info(f"Загрузка модели эмбеддингов: {EMB_MODEL_NAME} (device={DEVICE})")
emb_model = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)

logger.info(f"Загрузка модели суммаризации: {SUM_MODEL_NAME}")
sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME).to(DEVICE)

logger.info(f"Загрузка модели генерации рекомендаций: {REC_MODEL_NAME}")
rec_tokenizer = AutoTokenizer.from_pretrained(REC_MODEL_NAME)
rec_model = AutoModelForSeq2SeqLM.from_pretrained(REC_MODEL_NAME).to(DEVICE)

logger.info(f"Загрузка TTS-модели: {TTS_MODEL_NAME}")
tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=(DEVICE == "cuda"))

logger.info("Все модели успешно загружены.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pydantic-модели запросов/ответов
# ─────────────────────────────────────────────────────────────────────────────

class AddDocRequest(BaseModel):
    doc_id: str                         # уникальный идентификатор, напр. "ivan_2025-05-31_07-00"
    document: str                       # текст события, напр. "Иван в 07:00 варит кофе"
    embedding: Optional[List[float]]    # если None, генерируем сами
    metadata: Dict                      # любые метаданные, напр. {"user_id":"ivan","type":"raw_routine","timestamp":"2025-05-31T07:00:00"}

class QueryRequest(BaseModel):
    query_embedding: List[float]
    n_results: Optional[int] = 5

class QueryResultItem(BaseModel):
    doc_id: str
    document: str
    distance: float
    metadata: Dict

class QueryResponse(BaseModel):
    results: List[QueryResultItem]

class SummarizeRequest(BaseModel):
    texts: List[str]

class SummarizeResponse(BaseModel):
    summary: str

class RecommendRequest(BaseModel):
    profile_summary: str
    context: str

class RecommendResponse(BaseModel):
    recommendation: str

class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    audio_base64: str

class ChatRequest(BaseModel):
    history: List[Dict]        # [{"role":"user"|"assistant", "text":"..."}]
    user_input: str

class ChatResponse(BaseModel):
    response: str

# ─────────────────────────────────────────────────────────────────────────────
# 5. FastAPI-приложение
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartHome FastAPI Service",
    description="Всё в одном: ChromaDB + эмбеддинги + суммаризация + рекомендации + TTS + чат",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. УТИЛИТЫ: вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

async def generate_embedding(text: str) -> List[float]:
    """
    Вычисляем embedding для одного текста через SentenceTransformer.
    """
    emb = emb_model.encode([text], convert_to_numpy=True)
    return emb[0].tolist()

def summarize_texts(texts: List[str]) -> str:
    """
    Склеиваем список текстов, передаём в flan-t5-small с префиксом "summarize:" и возвращаем summary.
    Если texts слишком длинный, можно разбить на чанки (по 3-5 штук) и суммаризовать по частям.
    """
    joined = " [SEP] ".join(texts)
    prompt = "summarize: " + joined
    inputs = sum_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)
    summary_ids = sum_model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False
    )
    summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_recommendation(profile_summary: str, context: str) -> str:
    """
    Генерируем текст рекомендации на flan-t5-small.
    """
    prompt = (
        f"Профиль пользователя: {profile_summary}\n"
        f"Контекст: {context}\n"
        "Задача: сформулируй короткую дружелюбную рекомендацию или вопрос "
        "о том, запускать ли привычную рутину. Ответ дай на русском.\n"
        "Ответ:"
    )
    inputs = rec_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)
    out_ids = rec_model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )
    result = rec_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return result

def synthesize_speech(text: str) -> bytes:
    """
    Генерируем WAV-байты из текста через Coqui TTS. Сохраняем во временный файл, читаем и возвращаем.
    """
    tmp_path = os.path.join(os.getcwd(), "temp_tts.wav")
    tts.tts_to_file(text=text, file_path=tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data

def wav_to_base64(wav_bytes: bytes) -> str:
    """
    Кодирует WAV-байты в base64-строку.
    """
    return base64.b64encode(wav_bytes).decode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# 7. HTTP-РОУТЫ
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Проверка состояния сервера")
async def health_check():
    return {
        "status": "alive",
        "chroma_data_dir": CHROMADB_PERSIST,
        "device": DEVICE
    }

@app.post("/add", summary="Добавить или обновить документ в ChromaDB")
async def add_document(req: AddDocRequest):
    """
    Записываем документ в ChromaDB.
    Если req.embedding == None, генерируем embedding автоматически.
    """
    try:
        if req.embedding is None:
            emb = await generate_embedding(req.document)
        else:
            emb = req.embedding

        collection.add(
            ids=[req.doc_id],
            documents=[req.document],
            embeddings=[emb],
            metadatas=[req.metadata]
        )
    except Exception as e:
        logger.error(f"Ошибка в /add: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при добавлении документа: {e}")

    return {"status": "ok", "doc_id": req.doc_id}

@app.post("/query", response_model=QueryResponse, summary="Поиск ближайших документов в ChromaDB")
async def query_documents(req: QueryRequest):
    """
    Ищем ближайшие n_results документов по переданному embedding.
    """
    try:
        resp = collection.query(
            query_embeddings=[req.query_embedding],
            n_results=req.n_results
        )
    except Exception as e:
        logger.error(f"Ошибка в /query: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при поиске: {e}")

    results: List[QueryResultItem] = []
    ids = resp["ids"][0]
    docs = resp["documents"][0]
    dists = resp["distances"][0]
    metas = resp["metadatas"][0]

    for i, did in enumerate(ids):
        results.append(
            QueryResultItem(
                doc_id=did,
                document=docs[i],
                distance=dists[i],
                metadata=metas[i]
            )
        )
    return QueryResponse(results=results)

@app.post("/summarize", response_model=SummarizeResponse, summary="Суммаризация списка текстов")
async def summarize(req: SummarizeRequest):
    """
    Принимает {"texts": [строка1, строка2, ...]}, возвращает {"summary": "..."}.
    """
    try:
        summary = summarize_texts(req.texts)
    except Exception as e:
        logger.error(f"Ошибка в /summarize: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при суммаризации: {e}")

    return SummarizeResponse(summary=summary)

@app.post("/recommend", response_model=RecommendResponse, summary="Генерация рекомендации")
async def recommend(req: RecommendRequest):
    """
    Принимает {"profile_summary": "...", "context": "..."}, возвращает {"recommendation": "..."}.
    """
    try:
        rec = generate_recommendation(req.profile_summary, req.context)
    except Exception as e:
        logger.error(f"Ошибка в /recommend: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации рекомендации: {e}")

    return RecommendResponse(recommendation=rec)

@app.post("/tts", response_model=TTSResponse, summary="Синтез речи (TTS)")
async def tts_endpoint(req: TTSRequest):
    """
    Принимает {"text": "..."} и возвращает {"audio_base64": "..."}.
    """
    try:
        wav_bytes = synthesize_speech(req.text)
        b64 = wav_to_base64(wav_bytes)
    except Exception as e:
        logger.error(f"Ошибка в /tts: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при синтезе речи: {e}")

    return TTSResponse(audio_base64=b64)

@app.post("/chat", response_model=ChatResponse, summary="Простой чат-режим")
async def chat(req: ChatRequest):
    """
    Если нужно реализовать общий чат: берём history + user_input, строим prompt, отвечаем через flan-t5-small.
    """
    try:
        # Собираем историю чата
        parts = []
        for msg in req.history:
            role = msg.get("role", "user")
            text = msg.get("text", "")
            if role == "assistant":
                parts.append(f"Ассистент: {text}")
            else:
                parts.append(f"Пользователь: {text}")
        parts.append(f"Пользователь: {req.user_input}")
        prompt = "\n".join(parts) + "\nАссистент:"

        inputs = rec_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)
        out_ids = rec_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )
        response = rec_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Ошибка в /chat: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации чат-ответа: {e}")

    return ChatResponse(response=response)

# ─────────────────────────────────────────────────────────────────────────────
# 8. ЗАПУСК UVICORN (если запустить как python server.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn, sys
    logger.info("Запускаем Uvicorn server:app …")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # В продакшне False, при разработке можно True
    )
