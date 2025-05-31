import torchaudio
from vosk import Model as VoskModel, KaldiRecognizer
from fastapi import HTTPException

# Инициализация модели Vosk отдельно не показана (можно добавить по аналогии)


def recognize_speech_from_wav(wav_path: str) -> str:
    waveform, sample_rate = torchaudio.load(wav_path)

    if sample_rate != 16000:
        raise HTTPException(
            status_code=400,
            detail=f"STT поддерживает только WAV с частотой 16000 Гц. Ваше: {sample_rate} Гц"
        )

    # Здесь должен быть код распознавания через silero или Vosk
    # Например, используя vosk:
    model = VoskModel("path/to/vosk/model")
    rec = KaldiRecognizer(model, sample_rate)

    with open(wav_path, "rb") as wf:
        data = wf.read()
        rec.AcceptWaveform(data)
    result = json.loads(rec.FinalResult())
    return result.get("text", "")