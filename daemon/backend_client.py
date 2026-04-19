"""
HTTP client wrappers for the three back-end shims:
  - :8002 Qwen3-ASR    → transcribe(audio_bytes) → text
  - :8004 Hermes shim  → chat(text, session_id) → reply_text
  - :8003 VoxCPM2 TTS  → synth_stream(text) → async chunk iterator (48kHz int16 PCM)
"""
from __future__ import annotations

import io
import json
import logging
import os
import wave
from typing import Any, AsyncIterator

import httpx

import sys
from pathlib import Path as _Path
# Dynamically resolve repo root so the daemon works regardless of install path.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import config as cfg

from daemon import events

logger = logging.getLogger("backend-client")

# Long-lived HTTP client — connection pooling saves 10-30ms per call.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0),
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# ---------- ASR ----------
async def transcribe(audio_pcm_16k_mono: bytes, *, language: str | None = None) -> tuple[str, str | None]:
    """Pack int16 PCM @ 16kHz into a wav and POST to :8002.
    Returns (text, language). Empty text means ASR hallucinated or silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(cfg.MIC_SAMPLE_RATE)
        w.writeframes(audio_pcm_16k_mono)
    buf.seek(0)

    files = {"file": ("command.wav", buf.getvalue(), "audio/wav")}
    data = {}
    if language:
        data["language"] = language

    r = await _get_client().post(
        f"{cfg.QWEN3_ASR_URL}/v1/audio/transcriptions",
        files=files,
        data=data,
    )
    r.raise_for_status()
    j = r.json()
    return (j.get("text") or "").strip(), j.get("language")


# ---------- Hermes LLM (deep / agentic) ----------
async def chat(
    user_text: str,
    *,
    session_id: str,
    system_prompt: str | None = None,
    timeout_s: int | None = None,
) -> str:
    """POST to hermes-shim. Uses X-Session-Id for --resume continuity in Hermes."""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_text})

    read_timeout = timeout_s or cfg.HERMES_TIMEOUT_S

    r = await _get_client().post(
        f"{cfg.HERMES_SHIM_URL}/v1/chat/completions",
        json={"messages": msgs},
        headers={"X-Session-Id": session_id},
        timeout=httpx.Timeout(connect=5.0, read=read_timeout, write=10.0, pool=5.0),
    )
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()


# ---------- Subconscious (fast direct-to-SGLang) ----------
# Stream mode controls whether we use `stream=True` against SGLang. Streaming
# lets the TUI display tokens as they arrive, but we preserve the existing
# non-stream code path as fallback (env `SUBCONSCIOUS_STREAM=0`). We accumulate
# stream chunks into the same {choices[0].message.content/tool_calls} shape
# the caller already expects, so subconscious.chat() branching code is
# untouched. tool_calls typically arrive in the final chunk as a complete
# structure — we just pick up whichever chunks contain them.
_SUBCONSCIOUS_STREAM = os.environ.get("SUBCONSCIOUS_STREAM", "1") not in ("0", "false", "False")


async def _subconscious_chat_nonstream(body: dict, headers: dict) -> dict:
    r = await _get_client().post(
        f"{cfg.SGLANG_URL}/v1/chat/completions",
        json=body,
        headers=headers,
        timeout=httpx.Timeout(connect=5.0, read=45.0, write=10.0, pool=5.0),
    )
    r.raise_for_status()
    return r.json()


def _merge_tool_call_delta(agg: dict[int, dict], deltas: list[dict]) -> None:
    """OpenAI-style streaming tool_calls arrive as deltas indexed by
    `index`: the first delta contains id + function.name, subsequent
    deltas append characters to function.arguments. We merge into the
    `agg` dict keyed by index, preserving first-seen id/type/name and
    concatenating argument strings."""
    for d in deltas:
        idx = d.get("index", 0)
        slot = agg.setdefault(idx, {
            "id": None,
            "type": "function",
            "function": {"name": None, "arguments": ""},
        })
        # Merge id / type — keep first non-None
        if slot.get("id") is None and d.get("id"):
            slot["id"] = d["id"]
        if d.get("type"):
            slot["type"] = d["type"]
        fn_delta = d.get("function") or {}
        if fn_delta.get("name") and slot["function"]["name"] is None:
            slot["function"]["name"] = fn_delta["name"]
        arg_delta = fn_delta.get("arguments")
        if isinstance(arg_delta, str) and arg_delta:
            slot["function"]["arguments"] += arg_delta


async def _subconscious_chat_stream(body: dict, headers: dict, event_hop: int) -> dict:
    """Stream path — emits `sub_token` events for the TUI while aggregating
    chunks into the same response dict shape the non-stream caller expects.

    Two aggregation modes for tool_calls:
      1. Delta-style (OpenAI std): multiple chunks, merged by `index`
         (name in first chunk, arguments chars concat across subsequent).
      2. One-shot: single chunk with the complete array — also handled
         (the per-delta merger is idempotent for a single pass).
    """
    body = {**body, "stream": True}
    content_buf: list[str] = []
    tool_calls_agg: dict[int, dict] = {}
    final_finish_reason: str | None = None

    async with _get_client().stream(
        "POST",
        f"{cfg.SGLANG_URL}/v1/chat/completions",
        json=body,
        headers=headers,
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
    ) as r:
        r.raise_for_status()
        async for raw_line in r.aiter_lines():
            if not raw_line:
                continue
            line = raw_line.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                chunk = json.loads(line)
            except Exception:
                logger.debug("stream: unparseable chunk %r", line[:200])
                continue
            try:
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta") or {}
                fr = choice.get("finish_reason")
                if fr:
                    final_finish_reason = fr
                piece = delta.get("content")
                if piece:
                    content_buf.append(piece)
                    events.log("sub_token", delta=piece, hop=event_hop)
                tc = delta.get("tool_calls")
                if tc:
                    _merge_tool_call_delta(tool_calls_agg, tc)
            except Exception:
                logger.debug("stream: chunk shape unexpected %r", chunk)

    content = "".join(content_buf)
    message: dict[str, Any] = {"role": "assistant"}
    if content:
        message["content"] = content
    if tool_calls_agg:
        # Emit in index order for stable ordering.
        message["tool_calls"] = [tool_calls_agg[i] for i in sorted(tool_calls_agg.keys())]

    return {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": final_finish_reason or "stop",
            }
        ]
    }


async def subconscious_chat(
    messages: list[dict],
    *,
    tools: list[dict] | None = None,
    event_hop: int = 0,
) -> dict:
    """POST to SGLang /v1/chat/completions. Returns the full response dict
    regardless of stream vs non-stream mode. Caller logic unchanged."""
    headers = {"Content-Type": "application/json"}
    if cfg.SGLANG_API_KEY:
        headers["Authorization"] = f"Bearer {cfg.SGLANG_API_KEY}"

    body = {
        "model": cfg.SGLANG_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    if _SUBCONSCIOUS_STREAM:
        try:
            return await _subconscious_chat_stream(body, headers, event_hop)
        except Exception:
            # Fallback silently to non-stream on any error — we don't want
            # the TUI optional path to break the voice assistant.
            logger.exception("stream path failed; falling back to non-stream")
    return await _subconscious_chat_nonstream(body, headers)


# ---------- TTS ----------
import os as _os  # avoid shadowing


async def synth_stream(text: str) -> AsyncIterator[bytes]:
    """Stream 48kHz int16 LE mono PCM chunks from :8003/tts.
    Yields bytes as they arrive so the audio player can start playback before
    generation completes.

    If `cfg.VOXCPM_REFERENCE_WAV` points to an existing file, pass it as
    `reference_wav_path` so every utterance uses the same cloned voice
    (JARVIS Paul Bettany by default)."""
    body = {"text": text}
    if cfg.VOXCPM_REFERENCE_WAV and _os.path.exists(cfg.VOXCPM_REFERENCE_WAV):
        body["reference_wav_path"] = cfg.VOXCPM_REFERENCE_WAV

    async with _get_client().stream(
        "POST",
        f"{cfg.VOXCPM_URL}/tts",
        json=body,
        timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
    ) as r:
        r.raise_for_status()
        async for chunk in r.aiter_bytes(chunk_size=4096):
            if chunk:
                yield chunk
