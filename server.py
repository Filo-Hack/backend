# server.py

import os
import sys
import asyncio
import base64
import tempfile
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

import chromadb
from chromadb.config import Settings

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from TTS.api import TTS

from vosk import Model as VoskModel, KaldiRecognizer
import wave
import json

# ─────────────────────────────────────────────────────────────────────────────
# 1. НАСТРОЙКИ И ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ
# ─────────────────────────────────────────────────────────────────────────────

# Директория для хранения ChromaDB
CHROMADB_PERSIST = os.getenv(
    "CHROMADB_PERSIST_DIR",
    os.path.join(os.getcwd(), "chroma_data")
)

# Путь к модели Vosk
VOSK_MODEL_PATH = os.getenv(
    "VOSK_MODEL_PATH",
    os.path.join(os.getcwd(), "models", "vosk-ru")
)

# Имена моделей
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUM_MODEL_NAME = "google/flan-t5-small"
REC_MODEL_NAME = "google/flan-t5-small"
TTS_MODEL_NAME = "tts_models/ru/ru-Rus/VITS"

# Устройство для PyTorch ("cuda" если есть GPU, иначе "cpu")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# 2. ИНИЦИАЛИЗАЦИЯ ЛОГИРОВАНИЯ И ПАПОК
# ─────────────────────────────────────────────────────────────────────────────

# Создаём директорию для ChromaDB
os.makedirs(CHROMADB_PERSIST, exist_ok=True)
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

logger.info(f"Используем persist directory для ChromaDB: {CHROMADB_PERSIST}")
logger.info(f"Устройство для PyTorch: {DEVICE}")
logger.info(f"Путь к Vosk-модели: {VOSK_MODEL_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ИНИЦИАЛИЗАЦИЯ ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMADB_PERSIST
    )
)

try:
    collection = client.get_collection(name="user_profiles")
    logger.info("ChromaDB: коллекция 'user_profiles' найдена.")
except ValueError:
    collection = client.create_collection(name="user_profiles")
    logger.info("ChromaDB: коллекция 'user_profiles' создана.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. ЗАГРУЗКА МОДЕЛЕЙ (эмбеддинги, суммаризация, генерация, TTS, Vosk)
# ─────────────────────────────────────────────────────────────────────────────

# 4.1. SentenceTransformer для эмбеддингов
logger.info(f"Загрузка EMB-модели: {EMB_MODEL_NAME} (device={DEVICE})")
emb_model = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)

# 4.2. Flan-T5 для суммаризации
logger.info(f"Загрузка модели суммаризации: {SUM_MODEL_NAME}")
sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME).to(DEVICE)

# 4.3. Flan-T5 для рекомендаций / чата
logger.info(f"Загрузка модели рекомендаций/чата: {REC_MODEL_NAME}")
rec_tokenizer = AutoTokenizer.from_pretrained(REC_MODEL_NAME)
rec_model = AutoModelForSeq2SeqLM.from_pretrained(REC_MODEL_NAME).to(DEVICE)

# 4.4. Coqui TTS
logger.info(f"Загрузка TTS-модели: {TTS_MODEL_NAME}")
tts = TTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=(DEVICE == "cuda"))

# 4.5. Vosk (STT)
if not os.path.exists(VOSK_MODEL_PATH):
    logger.error(f"Модель Vosk не найдена по пути {VOSK_MODEL_PATH}. STT не будет доступен.")
    vosk_model = None
else:
    logger.info("Загрузка Vosk-модели …")
    vosk_model = VoskModel(VOSK_MODEL_PATH)
    logger.info("Vosk-модель успешно загружена.")

logger.info("Все модели успешно загружены.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Pydantic-модели запросов/ответов
# ─────────────────────────────────────────────────────────────────────────────

class AddDocRequest(BaseModel):
    doc_id: str                         # уникальный идентификатор, напр. "ivan_2025-05-31_07-00"
    document: str                       # текст события, напр. "Иван в 07:00 варит кофе"
    embedding: Optional[List[float]]    # если None, генерируем сами
    metadata: Dict                      # любые метаданные

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

class STTResponse(BaseModel):
    text: str

class ChatRequest(BaseModel):
    history: List[Dict]        # [{"role":"user"/"assistant", "text":"..."}]
    user_input: str

class ChatResponse(BaseModel):
    response: str

# ─────────────────────────────────────────────────────────────────────────────
# 6. FastAPI-приложение
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartHome FastAPI Service",
    description="Всё в одном: ChromaDB + эмбеддинги + суммаризация + рекомендации + TTS + STT + чат",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. УТИЛИТЫ: вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

async def generate_embedding(text: str) -> List[float]:
    emb = emb_model.encode([text], convert_to_numpy=True)
    return emb[0].tolist()

def summarize_texts(texts: List[str]) -> str:
    """
    1) Склеиваем список текстов, формируем промпт
    2) Передаём в flan-t5-small, получаем summary
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
    Генерация рекомендаций на flan-t5-small.
    """
    prompt = (
        f"Профиль пользователя: {profile_summary}\n"
        f"Контекст: {context}\n"
        "Задача: сформулируй короткую дружелюбную рекомендацию "
        "или вопрос о том, запускать ли привычную рутину. Ответ дай на русском.\n"
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
    Генерируем WAV-байты через Coqui TTS и возвращаем.
    """
    # Используем временный файл, чтобы избежать гонки имён:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    tts.tts_to_file(text=text, file_path=tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data

def wav_to_base64(wav_bytes: bytes) -> str:
    """
    Кодируем WAV-байты в base64.
    """
    return base64.b64encode(wav_bytes).decode("utf-8")

def recognize_speech_from_wav(wav_path: str) -> str:
    """
    Распознаём речь из WAV-файла с помощью Vosk.
    Предполагаем: WAV 16kHz, mono, PCM 16-bit.
    """
    if vosk_model is None:
        raise RuntimeError("Vosk-модель не загружена, STT недоступен.")
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in (8000, 16000, 48000):
        # Если формат не тот, можно предварительно перекодировать через ffmpeg или уведомить пользователя.
        raise HTTPException(
            status_code=400,
            detail=(
                f"Поддерживаются только WAV PCM 16-бит mono, частота 8k/16k/48k. "
                f"Ваши параметры: channels={wf.getnchannels()}, "
                f"samplewidth={wf.getsampwidth()}, framerate={wf.getframerate()}"
            )
        )
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    rec.SetWords(False)

    result_text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = rec.Result()
            text_json = json.loads(res)
            result_text += text_json.get("text", "") + " "
    # финальный кусок
    final = rec.FinalResult()
    text_json = json.loads(final)
    result_text += text_json.get("text", "")
    wf.close()
    return result_text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 8. HTTP-РОУТЫ
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", summary="Проверка состояния сервера")
async def health_check():
    return {
        "status": "alive",
        "chroma_data_dir": CHROMADB_PERSIST,
        "device": DEVICE,
        "vosk_model_loaded": bool(vosk_model),
    }

@app.post("/add", summary="Добавить или обновить документ в ChromaDB")
async def add_document(req: AddDocRequest):
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

@app.post("/stt", response_model=STTResponse, summary="Распознавание речи из WAV-аудио (STT)")
async def stt_endpoint(file: UploadFile = File(...)):
    """
    Принимает multipart/form-data с одним полем 'file' (WAV-аудио PCM 16-бит,mono).
    Возвращает JSON { "text": "распознанный текст" }.
    """
    if vosk_model is None:
        raise HTTPException(status_code=503, detail="STT недоступен: модель Vosk не загружена.")

    # Сохраняем WAV во временный файл
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)
    except Exception as e:
        logger.error(f"Ошибка при сохранении полученного файла: {e}")
        raise HTTPException(status_code=500, detail="Не удалось сохранить загруженный файл.")

    # Распознаём
    try:
        text = recognize_speech_from_wav(tmp_path)
    except HTTPException as he:
        os.remove(tmp_path)
        raise he
    except Exception as e:
        os.remove(tmp_path)
        logger.error(f"Ошибка в STT-модуле: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при распознавании речи: {e}")

    # Удаляем временный файл
    os.remove(tmp_path)
    return STTResponse(text=text)

@app.post("/chat", response_model=ChatResponse, summary="Простой чат-режим")
async def chat(req: ChatRequest):
    """
    Берём history + user_input, строим prompt, отвечаем через flan-t5-small.
    """
    try:
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
# 9. ЗАПУСК UVICORN (если запускаем напрямую)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info("Запускаем Uvicorn server:app …")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # в продакшне False, в dev можно True
    )
