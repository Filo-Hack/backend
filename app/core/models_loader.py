import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import whisper
from .config import EMB_MODEL_NAME, SUM_MODEL_NAME, REC_MODEL_NAME, DEVICE
from loguru import logger

# Загрузка моделей

logger.info(f"Загрузка модели эмбеддингов: {EMB_MODEL_NAME} (device={DEVICE})")
emb_model = SentenceTransformer(EMB_MODEL_NAME, device=DEVICE)

logger.info(f"Загрузка модели суммаризации: {SUM_MODEL_NAME}")
sum_tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL_NAME)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL_NAME).to(DEVICE)

logger.info(f"Загрузка модели генерации рекомендаций/чата: {REC_MODEL_NAME}")
rec_tokenizer = AutoTokenizer.from_pretrained(REC_MODEL_NAME)
rec_model = AutoModelForSeq2SeqLM.from_pretrained(REC_MODEL_NAME).to(DEVICE)

logger.info("Загрузка Whisper-модели: tiny")
whisper_model = whisper.load_model("tiny", device=DEVICE)
logger.info("Whisper-модель загружена.")