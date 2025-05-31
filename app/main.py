from fastapi import FastAPI
from .core.logger import logger
from .routers import (
    health,
    documents,
    summarize,
    recommend,
    tts,
    stt,
    chat
)

app = FastAPI(
    title="SmartHome FastAPI Service",
    description="ChromaDB ≥0.4.0 + эмбеддинги + суммаризация + рекомендации + TTS + STT + чат",
    version="1.0.0"
)

# Подключаем роуты
app.include_router(health.router, prefix="")
app.include_router(documents.router, prefix="")
app.include_router(summarize.router, prefix="")
app.include_router(recommend.router, prefix="")
app.include_router(tts.router, prefix="")
app.include_router(stt.router, prefix="")
app.include_router(chat.router, prefix="")

if __name__ == "__main__":
    import uvicorn
    logger.info("Запускаем Uvicorn server:app …")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )