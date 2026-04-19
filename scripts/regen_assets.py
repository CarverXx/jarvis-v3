"""
Generate JARVIS v3 audio assets:
  - voxcpm_zai.wav  — "在" spoken by VoxCPM2 (served at :8003/tts/wav)
  - waiting_beep.wav — soft sine ping used during Hermes tool-call waits

Idempotent: skips files that already exist unless --force is passed.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import wave
from pathlib import Path

import httpx
import numpy as np

# Make `config.py` at repo root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] regen_assets: %(message)s",
)
logger = logging.getLogger("regen_assets")


def ensure_waiting_beep(path: str, force: bool = False):
    if os.path.exists(path) and not force:
        logger.info("waiting_beep exists, skip: %s", path)
        return
    # 0.3s soft 440Hz sine with hanning envelope at 48kHz mono int16.
    # Volume ~20% so it's a gentle "heartbeat" not a loud alarm.
    sr = 48000
    dur = 0.3
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
    wave_f32 = np.sin(2 * np.pi * 440.0 * t) * np.hanning(n).astype(np.float32) * 0.20
    pcm = (wave_f32 * 32767.0).astype(np.int16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    logger.info("waiting_beep generated: %s (%.2fs @ %d Hz)", path, dur, sr)


def ensure_voxcpm_zai(path: str, force: bool = False, text: str = "在"):
    if os.path.exists(path) and not force:
        logger.info("voxcpm_zai exists, skip: %s", path)
        return
    url = f"{cfg.VOXCPM_URL}/tts/wav"
    body = {"text": text}
    # Pass the JARVIS reference so the wake cue shares the same voice as
    # every other TTS utterance; otherwise VoxCPM uses voice-design mode
    # and "在" sounds random every time the asset is regenerated.
    if cfg.VOXCPM_REFERENCE_WAV and os.path.exists(cfg.VOXCPM_REFERENCE_WAV):
        body["reference_wav_path"] = cfg.VOXCPM_REFERENCE_WAV
        logger.info("using reference_wav_path=%s", cfg.VOXCPM_REFERENCE_WAV)
    logger.info("requesting VoxCPM /tts/wav for text=%r → %s", text, path)
    try:
        r = httpx.post(url, json=body, timeout=120.0)
        r.raise_for_status()
    except Exception as e:
        logger.error("voxcpm2-tts unavailable (%s); skipping voxcpm_zai.wav — "
                     "satellite will fall back to beep awake.wav", e)
        return
    if r.headers.get("content-type", "").startswith("audio/wav"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(r.content)
        logger.info("voxcpm_zai generated: %s (%d bytes)", path, len(r.content))
    else:
        logger.error("voxcpm2-tts returned unexpected content-type %r", r.headers.get("content-type"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="regenerate even if exists")
    parser.add_argument("--zai-text", default="在", help="text spoken for the awake cue")
    args = parser.parse_args()

    ensure_waiting_beep(cfg.WAITING_BEEP_WAV, force=args.force)
    ensure_voxcpm_zai(cfg.AWAKE_ZAI_WAV, force=args.force, text=args.zai_text)


if __name__ == "__main__":
    main()
