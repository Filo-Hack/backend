import pyttsx3
from pydub import AudioSegment
import os
import base64
import tempfile
from fastapi import HTTPException


def synthesize_speech(text: str) -> bytes:
    """
    Синтез речи через pyttsx3 + pydub (WAV в память).
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('voice', 'russian')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
        tmp_path = tf.name

    engine.save_to_file(text, tmp_path)
    engine.runAndWait()

    audio = AudioSegment.from_file(tmp_path, format="wav")
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    os.remove(tmp_path)
    return buffer.getvalue()


def wav_to_base64(wav_bytes: bytes) -> str:
    return base64.b64encode(wav_bytes).decode("utf-8")