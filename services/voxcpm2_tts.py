"""
voxcpm2_tts — FastAPI TTS service wrapping OpenBMB VoxCPM2 (2B diffusion AR).

Listens on :8003. Exposes:

  POST /tts          — JSON body {text, ...}; returns streaming audio/x-raw
                       (48kHz int16 PCM mono chunks, chunked transfer).
                       Daemon can play chunks as they arrive, reducing
                       perceived latency vs waiting for the full wav.

  POST /tts/wav      — Same but buffers everything and returns a single
                       audio/wav for debugging/curl.

Model is loaded once via asyncio.Lock at first request. BF16 on CUDA.

Adapted from jarvis-v2 server/jarvis/services/minicpm_service.py (FastAPI
lifespan + asyncio.Lock single-instance pattern).
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

# Dedicated single-worker pool so every model call hits the SAME OS thread.
# VoxCPM2 uses torch._inductor.cudagraph_trees which stores per-thread TLS
# state; if consecutive generate_streaming/next() calls land on different
# threads, torch asserts `_is_key_in_tls(attr_name)` and crashes.
_INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voxcpm-infer")

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger("voxcpm2-tts")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

MODEL_ID = os.environ.get("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
MODEL_LOCAL_PATH = os.environ.get(
    "VOXCPM_LOCAL_PATH",
    os.path.expanduser("~/models/VoxCPM2"),
)
LOAD_DENOISER = os.environ.get("VOXCPM_LOAD_DENOISER", "false").lower() == "true"

# VoxCPM2 native output is 48 kHz float32 mono (AudioVAE V2 super-resolution)
SAMPLE_RATE = 48000


# ---------- Model lazy-load ----------
_model: Any | None = None
_load_lock = asyncio.Lock()


def _sync_load_model():
    """Load VoxCPM2 — runs on the INFERENCE_EXECUTOR thread so the torch
    _inductor TLS (cudagraph_trees tree_manager_containers) is initialised
    in the same OS thread that will call forward later. Mixing init+call
    threads crashes in torch 2.11 with `assert _is_key_in_tls(attr_name)`.

    We also **disable torch._dynamo entirely** — VoxCPM has a
    `fullgraph=True` @torch.compile'd layer that hits recompile-limit
    failures under varied input shapes (observed 2026-04-19). The small
    perf loss (~5-10% TTS latency) is acceptable; correctness matters.
    """
    import torch
    import torch._inductor.config
    import torch._dynamo.config
    # Kill cudagraph trees (TLS bug) + dynamo fullgraph recompile limit.
    torch._inductor.config.triton.cudagraphs = False
    try:
        torch._inductor.config.triton.cudagraph_trees = False
    except AttributeError:
        pass
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 128
    try:
        torch._dynamo.config.recompile_limit = 128
    except AttributeError:
        pass
    logger.info("torch.compile + cudagraphs disabled; loading VoxCPM2 from %s (denoiser=%s)…",
                MODEL_LOCAL_PATH, LOAD_DENOISER)
    t0 = time.monotonic()
    from voxcpm import VoxCPM
    model_src = MODEL_LOCAL_PATH if os.path.isdir(MODEL_LOCAL_PATH) else MODEL_ID
    m = VoxCPM.from_pretrained(model_src, load_denoiser=LOAD_DENOISER)
    logger.info("VoxCPM2 loaded in %.1fs (sample_rate=%d)",
                time.monotonic() - t0,
                getattr(m.tts_model, "sample_rate", SAMPLE_RATE))
    return m


async def _get_model():
    global _model
    if _model is not None:
        return _model
    async with _load_lock:
        if _model is not None:
            return _model
        loop = asyncio.get_running_loop()
        _model = await loop.run_in_executor(_INFERENCE_EXECUTOR, _sync_load_model)
        return _model


# ---------- Request schema ----------
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    # Voice design prompt embedded in text per VoxCPM2 convention:
    #   "(A young woman, gentle voice)Hello"
    # Or use reference wav for cloning:
    reference_wav_path: str | None = None
    prompt_wav_path: str | None = None
    prompt_text: str | None = None
    cfg_value: float = 2.0
    inference_timesteps: int = 10


# Cap (attenuation-only): VoxCPM occasionally peaks above -1 dBFS on
# fricatives/consonants, causing audible clipping through the Poly Sync
# mixer. We only *attenuate* over-level chunks to keep ≤ -3 dBFS; we
# never boost quiet chunks because per-chunk boost in streaming mode
# would flatten the natural dynamic (inter-word gaps would be pushed up
# to match loud chunks, which sounds worse than the original problem).
# Pairing this with a clean voice reference is what actually fixes
# "音量忽大忽小"; this is just clip protection.
PEAK_CAP_DBFS = float(os.environ.get("VOXCPM_PEAK_CAP_DBFS", "-3.0"))
_PEAK_CAP_AMP = 10.0 ** (PEAK_CAP_DBFS / 20.0)  # ≈0.7079


def _np_to_pcm_int16_bytes(audio: np.ndarray) -> bytes:
    """Convert float32 [-1,1] ndarray → little-endian int16 PCM bytes,
    with **attenuation-only** peak cap at PEAK_CAP_DBFS (no boost)."""
    if audio.dtype != np.int16:
        f = audio.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(f))) if f.size else 0.0
        if peak > _PEAK_CAP_AMP:
            f = f * (_PEAK_CAP_AMP / peak)
        clipped = np.clip(f, -1.0, 1.0)
        audio = (clipped * 32767.0).astype(np.int16)
    return audio.tobytes()


# ---------- FastAPI app ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.isdir(MODEL_LOCAL_PATH):
        logger.warning("local model dir %s not found — will pull %s on first request",
                       MODEL_LOCAL_PATH, MODEL_ID)
    yield
    logger.info("shutting down voxcpm2-tts")


app = FastAPI(title="voxcpm2-tts", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "model_loaded": _model is not None,
        "model_path": MODEL_LOCAL_PATH,
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/tts")
async def tts_stream(req: TTSRequest):
    """Streaming TTS — returns raw 48kHz int16 LE mono PCM chunks via
    chunked transfer encoding. Client plays chunks as they arrive.

    Headers:
      Content-Type: audio/x-raw; rate=48000; format=S16LE; channels=1
      X-Sample-Rate: 48000
    """
    m = await _get_model()
    logger.info("stream tts: text=%r (ref=%s)", req.text[:80], bool(req.reference_wav_path))

    async def gen() -> AsyncIterator[bytes]:
        t0 = time.monotonic()
        first_chunk = True
        total_samples = 0
        try:
            # VoxCPM2 generate_streaming is SYNC iterator; wrap with run_in_executor.
            loop = asyncio.get_running_loop()
            kwargs = dict(
                text=req.text,
                cfg_value=req.cfg_value,
                inference_timesteps=req.inference_timesteps,
            )
            if req.reference_wav_path:
                kwargs["reference_wav_path"] = req.reference_wav_path
            if req.prompt_wav_path:
                kwargs["prompt_wav_path"] = req.prompt_wav_path
            if req.prompt_text:
                kwargs["prompt_text"] = req.prompt_text

            # Run the sync generator in a thread; fetch each chunk with run_in_executor.
            iterator = await loop.run_in_executor(_INFERENCE_EXECUTOR, lambda: m.generate_streaming(**kwargs))
            sentinel = object()
            while True:
                chunk = await loop.run_in_executor(_INFERENCE_EXECUTOR, lambda: next(iterator, sentinel))
                if chunk is sentinel:
                    break
                if first_chunk:
                    logger.info("first-chunk @ %.2fs", time.monotonic() - t0)
                    first_chunk = False
                pcm = _np_to_pcm_int16_bytes(np.asarray(chunk))
                total_samples += len(pcm) // 2
                yield pcm
        except Exception as e:
            logger.exception("streaming tts failed")
            # Yield nothing more — client sees truncated stream; log says why.
            return
        dur = total_samples / SAMPLE_RATE
        logger.info("tts done: %.2fs audio in %.2fs wall",
                    dur, time.monotonic() - t0)

    return StreamingResponse(
        gen(),
        media_type="audio/x-raw",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Format": "S16LE",
            "X-Channels": "1",
        },
    )


@app.post("/tts/wav")
async def tts_wav(req: TTSRequest):
    """Non-streaming variant: buffers the full utterance and returns WAV.
    Useful for curl debugging. Daemon should use /tts (streaming)."""
    m = await _get_model()
    logger.info("wav tts: text=%r", req.text[:80])

    loop = asyncio.get_running_loop()
    kwargs = dict(
        text=req.text,
        cfg_value=req.cfg_value,
        inference_timesteps=req.inference_timesteps,
    )
    if req.reference_wav_path:
        kwargs["reference_wav_path"] = req.reference_wav_path

    t0 = time.monotonic()
    try:
        wav: np.ndarray = await loop.run_in_executor(_INFERENCE_EXECUTOR, lambda: m.generate(**kwargs))
    except Exception as e:
        logger.exception("non-streaming tts failed")
        raise HTTPException(status_code=500, detail=f"tts failed: {e}")

    elapsed = time.monotonic() - t0
    pcm = _np_to_pcm_int16_bytes(np.asarray(wav))
    dur = len(pcm) / 2 / SAMPLE_RATE
    logger.info("tts wav: %.2fs audio in %.2fs wall", dur, elapsed)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
    return Response(content=buf.getvalue(), media_type="audio/wav")


if __name__ == "__main__":
    port = int(os.environ.get("VOXCPM_PORT", "8003"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
