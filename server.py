
import os
import sys
import asyncio
import base64
import tempfile
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from loguru import logger

import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

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

CHROMADB_PERSIST = os.getenv(
    "CHROMADB_PERSIST_DIR",
    os.path.join(os.getcwd(), "chroma_data")
)

VOSK_MODEL_PATH = os.getenv(
    "VOSK_MODEL_PATH",
    os.path.join(os.getcwd(), "models", "vosk-ru")
)

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUM_MODEL_NAME = "google/flan-t5-small"
REC_MODEL_NAME = "google/flan-t5-small"
TTS_MODEL_NAME = "tts_models/ru/ru-Rus/VITS"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# 2. ИНИЦИАЛИЗАЦИЯ ЛОГИРОВАНИЯ И ПАПОК
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(CHROMADB_PERSIST, exist_ok=True)
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

logger.info(f"Используем persist directory для ChromaDB: {CHROMADB_PERSIST}")
logger.info(f"Устройство для PyTorch: {DEVICE}")
logger.info(f"Путь к Vosk-модели: {VOSK_MODEL_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. ИНИЦИАЛИЗАЦИЯ ChromaDB (новый API ≥ 0.4.0)
# ─────────────────────────────────────────────────────────────────────────────

# Если у вас было "устаревшее" хранилище и данные можно удалить, удалите папку CHROMADB_PERSIST
# или просто укажите другой пустой каталог. Иначе придётся делать миграцию через chroma-migrate.

try:
    # Пример без особых дополнительных настроек:
    client = chromadb.PersistentClient(
        path=CHROMADB_PERSIST,
        settings=Settings(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    logger.info("ChromaDB: PersistentClient создан.")
except Exception as e:
    logger.error(f"Ошибка при создании PersistentClient: {e}")
    sys.exit(1)

# Создаём или получаем коллекцию "user_profiles"
try:
    collection = client.get_or_create_collection(name="user_profiles")
    logger.info("ChromaDB: коллекция 'user_profiles' готова.")
except Exception as e:
    logger.error(f"Ошибка при создании/получении коллекции: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. ЗАГРУЗКА МОДЕЛЕЙ (эмбеддинги, суммаризация, генерация, TTS, Vosk)
# ─────────────────────────────────────────────────────────────────────────────

logger.info(f"Загрузка модели эмбеддингов: {EMB_MODEL_NAME} (device={DEVICE})")
emb_model = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)

logger.info(f"Загрузка модели суммаризации: {SUM_MODEL_NAME}")
sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
sum_model     = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME).to(DEVICE)

logger.info(f"Загрузка модели генерации рекомендаций/чата: {REC_MODEL_NAME}")
rec_tokenizer = AutoTokenizer.from_pretrained(REC_MODEL_NAME)
rec_model     = AutoModelForSeq2SeqLM.from_pretrained(REC_MODEL_NAME).to(DEVICE)

logger.info(f"Загрузка TTS-модели: {TTS_MODEL_NAME}")
tts           = TTS(model_name=TTS_MODEL_NAME, progress_bar=False, gpu=(DEVICE=="cuda"))

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
    doc_id: str
    document: str
    embedding: Optional[List[float]]
    metadata: Dict

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
    history: List[Dict]
    user_input: str

class ChatResponse(BaseModel):
    response: str

# ─────────────────────────────────────────────────────────────────────────────
# 6. FastAPI-приложение
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartHome FastAPI Service",
    description="ChromaDB ≥0.4.0 + эмбеддинги + суммаризация + рекомендации + TTS + STT + чат",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. УТИЛИТЫ: вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

async def generate_embedding(text: str) -> List[float]:
    emb = emb_model.encode([text], convert_to_numpy=True)
    return emb[0].tolist()

def summarize_texts(texts: List[str]) -> str:
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
    return sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_recommendation(profile_summary: str, context: str) -> str:
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
    return rec_tokenizer.decode(out_ids[0], skip_special_tokens=True)

def synthesize_speech(text: str) -> bytes:
    """
    Синтезируем WAV через Coqui TTS (rus VITS) и возвращаем байты.
    """
    tmp_path = os.path.join(os.getcwd(), "temp_tts.wav")
    tts.tts_to_file(text=text, file_path=tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.remove(tmp_path)
    return data

def wav_to_base64(wav_bytes: bytes) -> str:
    return base64.b64encode(wav_bytes).decode("utf-8")

def recognize_speech_from_wav(wav_path: str) -> str:
    if vosk_model is None:
        raise RuntimeError("Vosk-модель не загружена, STT недоступен.")
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in (8000, 16000, 48000):
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

@app.post("/add", summary="Добавить/обновить документ в ChromaDB")
async def add_document(req: AddDocRequest):
    try:
        emb = req.embedding if req.embedding is not None else await generate_embedding(req.document)
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

@app.post("/query", response_model=QueryResponse, summary="Поиск ближайших документов")
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
    ids     = resp["ids"][0]
    docs    = resp["documents"][0]
    dists   = resp["distances"][0]
    metas   = resp["metadatas"][0]

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
    try:
        summary = summarize_texts(req.texts)
    except Exception as e:
        logger.error(f"Ошибка в /summarize: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при суммаризации: {e}")
    return SummarizeResponse(summary=summary)

@app.post("/recommend", response_model=RecommendResponse, summary="Генерация рекомендации")
async def recommend(req: RecommendRequest):
    try:
        rec = generate_recommendation(req.profile_summary, req.context)
    except Exception as e:
        logger.error(f"Ошибка в /recommend: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при генерации рекомендации: {e}")
    return RecommendResponse(recommendation=rec)

@app.post("/tts", response_model=TTSResponse, summary="Синтез речи (TTS)")
async def tts_endpoint(req: TTSRequest):
    try:
        wav_bytes = synthesize_speech(req.text)
        b64 = wav_to_base64(wav_bytes)
    except Exception as e:
        logger.error(f"Ошибка в /tts: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при синтезе речи: {e}")
    return TTSResponse(audio_base64=b64)

@app.post("/stt", response_model=STTResponse, summary="Распознавание речи из WAV (STT)")
async def stt_endpoint(file: UploadFile = File(...)):
    if vosk_model is None:
        raise HTTPException(status_code=503, detail="STT недоступен: модель Vosk не загружена.")
    try:
        suffix = os.path.splitext(file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)
    except Exception as e:
        logger.error(f"Ошибка при сохранении файла: {e}")
        raise HTTPException(status_code=500, detail="Не удалось сохранить загруженный файл.")
    try:
        text = recognize_speech_from_wav(tmp_path)
    except HTTPException as he:
        os.remove(tmp_path)
        raise he
    except Exception as e:
        os.remove(tmp_path)
        logger.error(f"Ошибка в STT-модуле: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при распознавании речи: {e}")
    os.remove(tmp_path)
    return STTResponse(text=text)

@app.post("/chat", response_model=ChatResponse, summary="Простой чат-режим")
async def chat(req: ChatRequest):
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
# 9. Запуск (если через `python server.py`)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info("Запускаем Uvicorn server:app …")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
