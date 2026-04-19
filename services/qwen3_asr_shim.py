"""
qwen3_asr_shim — OpenAI-compatible /v1/audio/transcriptions over
Qwen3-ASR-1.7B (transformers backend).

Listens on :8002. Loads the model once at startup (lazy via asyncio.Lock).
Accepts multipart upload (OpenAI Whisper API style) and returns the
transcribed text.

Transformers backend is used instead of the vLLM backend because:
- Half-duplex flow does one-shot transcription per utterance (streaming
  not required)
- One process is simpler to manage than vllm-qwen3-asr.service + shim
- VRAM ~6GB fits comfortably on the RTX PRO 6000 96GB

Adapted from jarvis-v2 server/jarvis/asr/whisper_asr.py (lazy-load pattern)
and services/minicpm_service.py (FastAPI lifespan / asyncio.Lock).

Custom-word post-processing: Qwen3-ASR has no native hotword mechanism, so
we regex-replace known mishears (贾维斯/加维斯/jarves/chavez/...) → "Hermes".
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger("qwen3-asr-shim")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MODEL_PATH = os.environ.get(
    "QWEN3_ASR_MODEL_PATH",
    os.path.expanduser("~/models/Qwen3-ASR-1.7B"),
)
DEVICE = os.environ.get("QWEN3_ASR_DEVICE", "cuda:0")
DTYPE = os.environ.get("QWEN3_ASR_DTYPE", "bfloat16")  # "bfloat16" | "float16"
TARGET_SR = 16000

# Custom-word replacement — runs on the raw transcript before returning.
# Extend as you hit new mishears in production logs.
# NOTE: these target WAKE-WORD name confusion. Do NOT rewrite content speech.
_WAKEWORD_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(jarv[iu]?[sz]?|jarves|jervi[sz]?|jav[ei]s|jarvi[eou]s|jarvise)\b", re.IGNORECASE), "Hermes"),
    (re.compile(r"\b(chav[eé]z|hav[aá]s|horses?)\b", re.IGNORECASE), "Hermes"),
    # English mishears of "Hey Jarvis" observed in Qwen3-ASR logs 2026-04-19
    (re.compile(r"\bhey[,.\s]+james\b", re.IGNORECASE), "你好 Hermes"),
    (re.compile(r"\b(pajamas|the\s+hermes|hi\s+hermes|hey\s+hermes)\b", re.IGNORECASE), "你好 Hermes"),
    # "I'm not James. I am." / "not James" — when Qwen3-ASR misreads 贾维斯-correction
    (re.compile(r"\bi'?m\s+not\s+james\b", re.IGNORECASE), "你好 Hermes"),
    # Chinese canonical transliterations of "Jarvis"
    (re.compile(r"[贾家加佳][维][斯丝司思]"), "Hermes"),
]


def _apply_wakeword_replacements(text: str) -> str:
    for pat, repl in _WAKEWORD_REPLACEMENTS:
        text = pat.sub(repl, text)
    return text


# ---------- Model lazy-load (asyncio.Lock gated) ----------
_model: Any | None = None
_load_lock = asyncio.Lock()


async def _get_model():
    global _model
    if _model is not None:
        return _model
    async with _load_lock:
        if _model is not None:
            return _model
        logger.info("loading Qwen3-ASR model from %s (device=%s dtype=%s)…",
                    MODEL_PATH, DEVICE, DTYPE)
        t0 = time.monotonic()
        # Import here so the server can import the module without touching
        # CUDA until the first request (faster cold-start test).
        import torch
        from qwen_asr import Qwen3ASRModel

        torch_dtype = getattr(torch, DTYPE)
        _model = Qwen3ASRModel.from_pretrained(
            MODEL_PATH,
            dtype=torch_dtype,
            device_map=DEVICE,
            max_inference_batch_size=1,   # single-request service, no batching
            max_new_tokens=256,
        )
        logger.info("Qwen3-ASR loaded in %.1fs", time.monotonic() - t0)
        return _model


# ---------- Audio decode ----------
def _decode_to_16k_mono(audio_bytes: bytes) -> np.ndarray:
    """Decode arbitrary-format audio bytes → 16kHz mono float32 ndarray.

    soundfile handles wav/flac/ogg/mp3 (via libsndfile). For webm/opus
    containers we'd need `av`/ffmpeg; daemon side should send wav/pcm.
    """
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"audio decode failed: {e}")
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono
    if sr != TARGET_SR:
        # Minimal resample via scipy.signal.resample_poly (installed)
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), TARGET_SR)
        up, down = TARGET_SR // g, int(sr) // g
        data = resample_poly(data, up, down).astype(np.float32)
    return data


# ---------- FastAPI app ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.isdir(MODEL_PATH):
        logger.error("model dir %s not found — requests will 500", MODEL_PATH)
    else:
        logger.info("model dir: %s (will load on first request)", MODEL_PATH)
    yield
    logger.info("shutting down qwen3-asr-shim")


app = FastAPI(title="qwen3-asr-shim", version="0.1.0", lifespan=lifespan)


class TranscriptionResponse(BaseModel):
    text: str
    language: str | None = None
    duration_s: float


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model_loaded": _model is not None,
        "model_path": MODEL_PATH,
        "device": DEVICE,
    }


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcriptions(
    file: UploadFile = File(...),
    model: str | None = Form(None),       # OpenAI field; ignored, we're fixed
    language: str | None = Form(None),    # "zh" / "en" / None=auto
    response_format: str | None = Form(None),  # only "json" supported
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    t0 = time.monotonic()
    pcm = _decode_to_16k_mono(raw)
    decode_s = time.monotonic() - t0

    m = await _get_model()

    # Map OpenAI ISO-639-1 codes to Qwen's expected names (None = auto)
    qwen_lang: str | None
    if not language:
        qwen_lang = None
    elif language.lower() in ("zh", "zh-cn", "chinese"):
        qwen_lang = "Chinese"
    elif language.lower() in ("en", "en-us", "english"):
        qwen_lang = "English"
    else:
        qwen_lang = language

    t1 = time.monotonic()
    try:
        results = m.transcribe(audio=(pcm, TARGET_SR), language=qwen_lang)
    except Exception as e:
        logger.exception("transcribe failed")
        raise HTTPException(status_code=500, detail=f"transcribe failed: {e}")
    infer_s = time.monotonic() - t1

    if not results:
        return TranscriptionResponse(text="", language=None, duration_s=len(pcm) / TARGET_SR)

    raw_text = (results[0].text or "").strip()
    text = _apply_wakeword_replacements(raw_text)
    lang = getattr(results[0], "language", None)
    duration = len(pcm) / TARGET_SR

    logger.info(
        "asr ok: decode=%.2fs infer=%.2fs dur=%.2fs lang=%s raw=%r → %r",
        decode_s, infer_s, duration, lang, raw_text[:80], text[:80],
    )
    return TranscriptionResponse(text=text, language=lang, duration_s=duration)


if __name__ == "__main__":
    port = int(os.environ.get("QWEN3_ASR_SHIM_PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
