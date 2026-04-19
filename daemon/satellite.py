"""
JARVIS v3 satellite — always-listening wake-word half-duplex voice daemon.

Architecture (abridged):
  PortAudio callback thread: reads 512-sample int16 chunks @ 16 kHz into a
  queue. Main asyncio loop polls the queue, drives a finite state machine:
      IDLE → LISTENING → PROCESSING → RESPONDING → FOLLOW_UP_LISTEN → IDLE
  openWakeWord gates IDLE→LISTENING, Silero VAD end-of-speech gates
  LISTENING→PROCESSING and speech-detect for FOLLOW_UP_LISTEN.

  PROCESSING does HTTP → qwen3-asr-shim → hermes-shim → voxcpm2-tts, the
  result streams into a blocking sounddevice OutputStream on the TTS
  executor thread. mic_muted flag drops frames during RESPONDING so the
  speakers don't loop back into wake word detection.

Reference: Wyoming Satellite `WakeStreamingSatellite`, OVOS `DinkumVoiceLoop`
(hybrid_listen), HA `AssistSatelliteState`.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import queue
import sys
import threading
import time
import uuid
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

# Make sibling `config.py` importable when running as `python daemon/satellite.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

from daemon.state import State
from daemon.wake_word import WakeWordDetector, FRAME_SAMPLES as WAKE_FRAME_SAMPLES
from daemon.vad import SileroVAD, CHUNK_SAMPLES as VAD_CHUNK_SAMPLES
from daemon import backend_client
from daemon.subconscious import Subconscious
from daemon import events

logger = logging.getLogger("satellite")

# System prompt injected at the TOP of every chat turn that goes through
# the Hermes CLI subprocess path (tool-call delegation). User-facing persona
# is JARVIS (贾维斯), matching the wake word. "Hermes" is ONLY an internal
# service/implementation name and must never surface in replies.
SYSTEM_PROMPT = (
    "你是贾维斯，家庭语音助手。"
    "只能用简体中文口语化回答，即使用户夹英文。"
    "**名字一律写中文『贾维斯』三个字，绝对不要写 JARVIS / J.A.R.V.I.S / Jarvis 等英文或字母拼写** — "
    "否则语音合成会一个字母一个字母念出来。"
    "回答简短（1-3 句能说完在 10 秒内），不要列表 / markdown / emoji / 读网址或长 JSON。"
    "不要解释自己是什么模型 / 什么架构，不要自我介绍除非被问。"
    "问名字就答『我是贾维斯』，不要说 Hermes / 潜意识 / 子系统等内部名词。"
    "如果输入像是 wake word 误识别（仅有 James / Pajamas / 贾维斯 等无实质问题），"
    "只说『在呢，请讲』即可。"
)

# ============================================================
# Audio device helper (PipeWire-preferred, Poly Sync fallback)
# ============================================================
def find_device(hint: str, kind: str) -> int | str | None:
    """kind='input'|'output'.

    Resolution order (learned 2026-04-19 during jarvis-v3 bring-up):
    1. PortAudio-enumerated name match for `hint` — works when PortAudio
       exposes the USB device directly.
    2. ALSA hw name via card id (e.g. 'hw:P10,0') — forces direct ALSA
       capture bypassing PulseAudio/PipeWire. Needed on this host because
       PipeWire's `pipewire` device routes to `aec_mic` (2ch float32 48kHz)
       whose down-conversion to 1ch 16kHz yields near-silence (-62 dBFS).
    3. `pipewire` as last fallback.
    """
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = sd.query_devices()

    if hint and hint.lower() not in ("", "default"):
        q = hint.lower()
        for i, d in enumerate(devices):
            if q in d["name"].lower() and d[channel_key] >= 1:
                logger.info("%s device → %r idx=%d (name match)", kind, d["name"], i)
                return i

    # Try ALSA hw names. Both the plain hw and the plug (auto-convert) form.
    # Poly Sync 10 registers as ALSA card "P10" on this host.
    for alsa_name in ("hw:P10,0", "plughw:P10,0", "sysdefault:CARD=P10"):
        try:
            info = sd.query_devices(alsa_name, kind=("input" if kind == "input" else "output"))
            if info[channel_key] >= 1:
                logger.info("%s device → %r (ALSA direct, ch=%d, sr=%.0f)",
                            kind, alsa_name, info[channel_key], info.get("default_samplerate", 0))
                return alsa_name
        except Exception as e:
            logger.debug("ALSA name %r unavailable: %s", alsa_name, e)

    for i, d in enumerate(devices):
        if d["name"].lower() == "pipewire" and d[channel_key] >= 1:
            logger.info("%s device → 'pipewire' idx=%d (last-resort fallback)", kind, i)
            return i

    logger.warning("no %s device match for hint=%r; using system default", kind, hint)
    return None


def load_asset_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a mono 16-bit WAV asset; returns (int16_samples, sample_rate)."""
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1, "asset must be mono"
        assert w.getsampwidth() == 2, "asset must be 16-bit"
        sr = w.getframerate()
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16), sr


# ============================================================
# Audio player — plays 48 kHz TTS chunks + assets; owns mic_muted flag
# ============================================================
class AudioPlayer:
    """Blocking OutputStream wrapped in an executor thread. Asyncio caller
    awaits play() futures. Holds the `mic_muted` threading.Event that
    MicReader checks before forwarding frames to wake/VAD."""

    def __init__(self, device: int | None):
        self.device = device
        self.mic_muted = threading.Event()
        self._lock = threading.Lock()
        # Track the current waiting-loop task (set by start_loop / cleared by stop_loop)
        self._loop_stop: asyncio.Event | None = None

    def _play_blocking(self, samples: np.ndarray, sample_rate: int):
        """Runs in executor thread. Plays given int16 mono samples synchronously."""
        try:
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                device=self.device,
            ) as stream:
                stream.write(samples)
        except Exception:
            logger.exception("audio playback failed")

    async def play_samples(self, samples: np.ndarray, sample_rate: int):
        """Queue one-shot playback; blocks caller until done."""
        self.mic_muted.set()
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._play_blocking, samples, sample_rate)
        finally:
            # Post-TTS grace so AEC tail doesn't re-trigger wake
            await asyncio.sleep(cfg.POST_TTS_GRACE_MS / 1000.0)
            self.mic_muted.clear()

    async def play_asset(self, path: str):
        """Play a prompt sound (awake.wav / error.wav). Honours mic_muted."""
        samples, sr = load_asset_wav(path)
        await self.play_samples(samples, sr)

    async def start_loop(self, wav_path: str, interval_s: float):
        """Start a heartbeat-style waiting loop: plays wav_path every
        interval_s seconds until stop_loop is awaited. mic_muted is held
        True for the entire duration so ambient + echo cannot re-trigger
        wake. Returns a Task handle to pass back into stop_loop."""
        samples, sr = load_asset_wav(wav_path)
        self._loop_stop = asyncio.Event()
        self.mic_muted.set()

        async def _pulse():
            loop = asyncio.get_running_loop()
            try:
                while not self._loop_stop.is_set():
                    t0 = time.monotonic()
                    await loop.run_in_executor(None, self._play_blocking, samples, sr)
                    elapsed = time.monotonic() - t0
                    # Wait the remainder of the interval (or exit if stopped).
                    remaining = max(0.05, interval_s - elapsed)
                    try:
                        await asyncio.wait_for(
                            self._loop_stop.wait(), timeout=remaining
                        )
                        break  # stop_loop fired
                    except asyncio.TimeoutError:
                        continue
            finally:
                pass  # mic_muted cleared by stop_loop

        task = asyncio.create_task(_pulse(), name="waiting-loop")
        logger.info("waiting-loop started (wav=%s every %.1fs)",
                    Path(wav_path).name, interval_s)
        return task

    async def stop_loop(self, task):
        """Stop a running waiting loop and release mic_muted."""
        if self._loop_stop is not None:
            self._loop_stop.set()
        try:
            await asyncio.wait_for(task, timeout=2.0)
        except asyncio.TimeoutError:
            task.cancel()
            logger.warning("waiting-loop cancel after timeout")
        except Exception:
            logger.exception("waiting-loop join error")
        # Short grace before unmuting
        await asyncio.sleep(cfg.POST_TTS_GRACE_MS / 1000.0)
        self.mic_muted.clear()
        self._loop_stop = None
        logger.info("waiting-loop stopped")

    async def play_tts_stream(self, chunk_iter, sample_rate: int):
        """Play a streaming TTS generator (async iterator of int16 PCM bytes)
        chunk-by-chunk. This keeps start-of-speech latency low: playback
        begins on the first chunk, not after full synthesis.
        """
        self.mic_muted.set()
        loop = asyncio.get_running_loop()
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            device=self.device,
        )
        stream.start()
        t0 = time.monotonic()
        bytes_played = 0
        last_tick = t0
        try:
            async for chunk in chunk_iter:
                if not chunk:
                    continue
                samples = np.frombuffer(chunk, dtype=np.int16)
                await loop.run_in_executor(None, stream.write, samples)
                bytes_played += len(chunk)
                now = time.monotonic()
                if now - last_tick >= 0.5:
                    events.log("tts_playing",
                               bytes_played=bytes_played,
                               elapsed_s=round(now - t0, 2))
                    last_tick = now
        finally:
            try:
                # Drain final bytes before stop
                await loop.run_in_executor(None, stream.stop)
                await loop.run_in_executor(None, stream.close)
            except Exception:
                pass
            await asyncio.sleep(cfg.POST_TTS_GRACE_MS / 1000.0)
            self.mic_muted.clear()


# ============================================================
# Mic reader — parec subprocess → reader thread → asyncio.Queue
# ============================================================
# Why parec and not sounddevice? PortAudio's PulseAudio backend on this host
# routes the "pipewire" device to the PipeWire default source (aec_mic,
# 2ch float32 48kHz). Down-sampling that to 1ch int16 16kHz lands on
# near-silence (RMS -62 dBFS) because the AEC side-tone layout isn't a plain
# mono capture. Direct ALSA hw:P10,0 can't be opened through PortAudio either
# (PulseAudio intercepts). parec pipes raw s16le 1ch 16kHz from the Poly Sync
# mono-fallback source straight to us — verified: -35 dBFS with speech,
# peak 0.2, which is what openWakeWord needs.
import subprocess


class MicReader:
    """Spawns `parec -d <source>` reading s16le mono @ 16 kHz. A background
    thread reads 1024 bytes (512 int16 samples = 32 ms) at a time and forwards
    to the asyncio.Queue. Honours mic_muted (drops frames during TTS)."""

    # Mic source for `parec` capture. Find your device name with:
    #   pactl list sources short | grep -v monitor
    # and set MIC_PAREC_SOURCE env var accordingly.
    PAREC_SOURCE = os.environ.get(
        "MIC_PAREC_SOURCE",
        "alsa_input",  # generic placeholder — user MUST override
    )
    CHUNK_BYTES = VAD_CHUNK_SAMPLES * 2  # 512 samples × 2 bytes (int16 LE)

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        out_queue: asyncio.Queue,
        mic_muted: threading.Event,
    ):
        self.loop = loop
        self.out_q = out_queue
        self.mic_muted = mic_muted
        self.proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _reader_thread(self):
        """Blocking read from parec stdout; push 512-sample chunks to queue.
        Applies MIC_INPUT_GAIN with soft-clip protection before forwarding."""
        assert self.proc and self.proc.stdout is not None
        stdout = self.proc.stdout
        gain = cfg.MIC_INPUT_GAIN
        while not self._stop.is_set():
            try:
                data = stdout.read(self.CHUNK_BYTES)
            except Exception:
                logger.exception("parec read error")
                break
            if not data:
                logger.warning("parec EOF — subprocess died?")
                break
            if len(data) < self.CHUNK_BYTES:
                # Short read at shutdown — ignore
                continue
            if self.mic_muted.is_set():
                continue
            chunk = np.frombuffer(data, dtype=np.int16)
            if gain != 1.0:
                # Upcast to float32 for headroom, apply gain, then clip back to int16.
                # np.clip is faster than tanh soft-clip and avoids distortion at
                # the typical speech levels we see (peak ≤ 0.2 pre-gain).
                boosted = chunk.astype(np.float32) * gain
                np.clip(boosted, -32768.0, 32767.0, out=boosted)
                chunk = boosted.astype(np.int16)
            self.loop.call_soon_threadsafe(_safe_put_nowait, self.out_q, chunk)

    def start(self):
        self.proc = subprocess.Popen(
            ["parec",
             "-d", self.PAREC_SOURCE,
             "--rate=16000",
             "--channels=1",
             "--format=s16le",
             "--raw",
             "--latency-msec=32"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._thread = threading.Thread(
            target=self._reader_thread, name="mic-parec-reader", daemon=True
        )
        self._thread.start()
        logger.info("mic reader: parec source=%s chunk_bytes=%d",
                    self.PAREC_SOURCE, self.CHUNK_BYTES)

    def stop(self):
        self._stop.set()
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


def _safe_put_nowait(q: asyncio.Queue, item):
    """Drop oldest if full instead of raising — we'd rather lose a frame
    than let the mic back-pressure the PortAudio callback."""
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
            q.put_nowait(item)
        except Exception:
            pass


# ============================================================
# Helper: concat int16 chunks into one np.ndarray / bytes
# ============================================================
def concat_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(chunks)


# ============================================================
# State machine driver
# ============================================================
class Satellite:
    def __init__(self):
        self.state: State = State.IDLE
        self.session_id: str = cfg.SESSION_ID_ENV or f"jarvis-v3-{uuid.uuid4().hex[:12]}"
        logger.info("session_id=%s", self.session_id)
        events.log("session", session_id=self.session_id)

        # Audio I/O — mic via parec subprocess (PortAudio can't reach Poly
        # Sync mono source on this host). Output still via sounddevice, which
        # routes fine to default sink.
        self.dev_out = find_device(cfg.SPEAKER_DEVICE_HINT, "output")

        # Models (CPU only)
        self.wake = WakeWordDetector(threshold=cfg.WAKE_THRESHOLD)
        self.vad = SileroVAD(threshold=cfg.VAD_WAKE_THRESHOLD)

        # Runtime
        self.player = AudioPlayer(device=self.dev_out)
        self.subconscious = Subconscious(session_id=self.session_id)

        # Populated in run()
        self.mic_q: asyncio.Queue | None = None
        self.mic: MicReader | None = None
        self._wake_acc = np.zeros(0, dtype=np.int16)

    # ---- helpers -------------------------------------------------------
    async def _drain_mic(self, seconds: float):
        """Throw away mic frames for `seconds` (e.g. after wake cue beep)."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            try:
                await asyncio.wait_for(self.mic_q.get(), timeout=0.1)
            except asyncio.TimeoutError:
                break

    async def _next_chunk(self, timeout: float) -> np.ndarray | None:
        try:
            return await asyncio.wait_for(self.mic_q.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    # ---- IDLE: wait for wake word -------------------------------------
    async def _run_idle(self) -> State:
        # Flush any stale mic frames left over from LISTENING/FOLLOW_UP/TTS-echo
        # (2026-04-19 "在 spam" bug root cause — residual speech tail in the
        # queue was being consumed by wake detector right after going IDLE).
        drained = 0
        while True:
            try:
                self.mic_q.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        # Cooldown: skip wake triggering for 1.5s after entering IDLE. This
        # covers the time window where openWakeWord's internal Mel buffer
        # still contains features from the just-finished TTS (JARVIS saying
        # "贾维斯" / "在" echoes back and fires wake on its own voice).
        wake_cooldown_end = time.monotonic() + 1.5
        logger.info("IDLE · listening for wake word (threshold=%.2f, mic_gain=%.1fx, drained=%d, cooldown=1.5s)",
                    cfg.WAKE_THRESHOLD, cfg.MIC_INPUT_GAIN, drained)
        events.log("state", to="IDLE", drained_frames=drained)
        self._wake_acc = np.zeros(0, dtype=np.int16)
        # Debug window: keep all wake scores in the last 5s, report top-3 +
        # near-miss count. This makes it easy to see whether the user's
        # "Hey Jarvis" was close (0.3-0.5) or totally missed (<0.1).
        window_scores: list[float] = []
        window_peak_rms = -90.0
        debug_next_log = time.monotonic() + 5.0
        chunk_count = 0
        # TUI pulse: emit a mic RMS event every ~8 chunks (~250ms)
        mic_pulse_every = 8
        mic_pulse_counter = 0
        while True:
            chunk = await self._next_chunk(timeout=1.0)
            if chunk is None:
                continue
            chunk_count += 1
            mic_pulse_counter += 1
            chunk_f = chunk.astype(np.float32) / 32768.0
            chunk_rms = 20.0 * np.log10(max(1e-9, float(np.sqrt(np.mean(chunk_f ** 2)))))
            if chunk_rms > window_peak_rms:
                window_peak_rms = chunk_rms
            if mic_pulse_counter >= mic_pulse_every:
                mic_pulse_counter = 0
                events.log("mic", rms_dbfs=round(chunk_rms, 1),
                           peak=round(float(np.max(np.abs(chunk_f))), 3))
            # Part 14 approach: feed EVERY chunk to openWakeWord. The acoustic
            # model itself is the speech gate; adding a Silero pre-filter risks
            # dropping short-phoneme wake tokens (Silero's 32ms window on
            # "hey" is ambiguous). VAD is still used for end-of-command.
            self._wake_acc = np.concatenate([self._wake_acc, chunk])
            while self._wake_acc.size >= WAKE_FRAME_SAMPLES:
                frame = self._wake_acc[:WAKE_FRAME_SAMPLES]
                self._wake_acc = self._wake_acc[WAKE_FRAME_SAMPLES:]
                triggered, score = self.wake.predict(frame)
                window_scores.append(float(score))
                # Emit wake events for the TUI whenever score is non-trivial.
                # Lowered 0.1→0.05 2026-04-19 so near-miss attempts show up.
                if score > 0.05 or triggered:
                    events.log("wake", score=round(float(score), 3),
                               triggered=bool(triggered))
                if triggered:
                    # Suppress triggers during post-TTS cooldown — JARVIS's
                    # own voice saying "贾维斯" leaks back and can fire wake.
                    if time.monotonic() < wake_cooldown_end:
                        remaining = wake_cooldown_end - time.monotonic()
                        logger.info("⏸ WAKE suppressed (score=%.3f) · cooldown %.1fs remaining",
                                    score, remaining)
                        continue
                    logger.info("🟢 WAKE triggered (score=%.3f ≥ %.2f)",
                                score, cfg.WAKE_THRESHOLD)
                    return State.LISTENING
            if time.monotonic() >= debug_next_log:
                if window_scores:
                    top3 = sorted(window_scores, reverse=True)[:3]
                    near_miss = sum(1 for s in window_scores if s >= 0.3)
                else:
                    top3 = [0.0]
                    near_miss = 0
                logger.info(
                    "IDLE heartbeat · chunks=%d top3_wake=[%s] near_miss≥0.3=%d peak_rms=%.1f dBFS",
                    chunk_count,
                    ", ".join(f"{s:.3f}" for s in top3),
                    near_miss,
                    window_peak_rms,
                )
                window_scores.clear()
                window_peak_rms = -90.0
                chunk_count = 0
                debug_next_log = time.monotonic() + 5.0

    # ---- LISTENING: capture command until VAD silence ------------------
    async def _run_listening(self, follow_up: bool = False) -> tuple[State, np.ndarray | None]:
        # Play awake prompt only on fresh wake (not follow-up — it'd be noisy)
        if not follow_up:
            # Priority: soft chime > VoxCPM "在" > plain beep.
            # Chime avoids the "repeats 在 every turn feels weird" issue.
            if os.path.exists(cfg.AWAKE_TONE_WAV):
                awake_path = cfg.AWAKE_TONE_WAV
            elif os.path.exists(cfg.AWAKE_ZAI_WAV):
                awake_path = cfg.AWAKE_ZAI_WAV
            else:
                awake_path = cfg.AWAKE_WAV
            await self.player.play_asset(awake_path)
            await self._drain_mic(0.25)  # drop AEC'd prompt echo

        logger.info("LISTENING · %scapturing command (hard timeout %.1fs, silence EOS %.2fs)",
                    "(follow-up) " if follow_up else "", cfg.LISTEN_HARD_TIMEOUT_S, cfg.EOS_SILENCE_S)
        events.log("state", to="LISTENING", follow_up=bool(follow_up))

        # Per-chunk silence-since-last-speech counter (in chunks of 32ms)
        silence_chunks_limit = max(1, int(cfg.EOS_SILENCE_S * 1000 / cfg.MIC_CHUNK_MS))
        min_speech_chunks = max(1, int(cfg.MIN_UTTERANCE_S * 1000 / cfg.MIC_CHUNK_MS))

        buf: list[np.ndarray] = []
        speech_chunks = 0
        silence_since_speech = 0
        started_speech = False
        start_t = time.monotonic()

        while True:
            if time.monotonic() - start_t > cfg.LISTEN_HARD_TIMEOUT_S:
                logger.info("LISTENING · hard timeout")
                break
            chunk = await self._next_chunk(timeout=0.5)
            if chunk is None:
                continue
            buf.append(chunk)
            is_sp, _ = self.vad.is_speech(chunk)
            if is_sp:
                started_speech = True
                speech_chunks += 1
                silence_since_speech = 0
            else:
                if started_speech:
                    silence_since_speech += 1
                    if silence_since_speech >= silence_chunks_limit:
                        break

        if not started_speech or speech_chunks < min_speech_chunks:
            logger.info("LISTENING · no/short utterance (speech_chunks=%d) → back to IDLE",
                        speech_chunks)
            return State.IDLE, None
        audio = concat_chunks(buf)
        logger.info("LISTENING · captured %.2fs (%d frames, %d speech chunks)",
                    len(audio) / cfg.MIC_SAMPLE_RATE, len(buf), speech_chunks)
        return State.PROCESSING, audio

    # ---- PROCESSING: ASR + Subconscious (delegates to Hermes tool on demand) -----
    async def _run_processing(self, audio: np.ndarray) -> tuple[State, str | None]:
        events.log("state", to="PROCESSING")
        pcm_bytes = audio.astype(np.int16).tobytes()
        dur_s = len(audio) / cfg.MIC_SAMPLE_RATE
        events.log("asr_start", duration_s=round(dur_s, 2))
        try:
            t0 = time.monotonic()
            text, lang = await backend_client.transcribe(pcm_bytes)
            elapsed = time.monotonic() - t0
            logger.info("PROCESSING · ASR %.2fs: %r (lang=%s)",
                        elapsed, text[:120], lang)
            events.log("asr_done", text=text, lang=lang or "",
                       elapsed_s=round(elapsed, 3))
        except Exception as e:
            logger.exception("ASR failed")
            events.log("error", where="asr", msg=str(e))
            return State.ERROR, f"ASR 失败: {e}"
        if not text:
            events.log("error", where="asr", msg="empty transcript")
            return State.ERROR, "没听清, 请再说一遍"

        # Route everything through Subconscious. It decides whether to answer
        # directly (fast path) or call invoke_hermes (deep path, auto-plays
        # the heartbeat waiting loop).
        try:
            t1 = time.monotonic()
            reply = await self.subconscious.chat(
                text,
                waiting_player=self.player,
            )
            logger.info("PROCESSING · Subconscious %.2fs: %r",
                        time.monotonic() - t1, reply[:120])
        except Exception as e:
            logger.exception("Subconscious failed")
            events.log("error", where="subconscious", msg=str(e))
            return State.ERROR, f"系统忙, 请稍候"
        if not reply:
            return State.ERROR, "我没想好, 请再说一遍"
        return State.RESPONDING, reply

    # ---- RESPONDING: streaming TTS ------------------------------------
    async def _run_responding(self, reply: str) -> State:
        # Sanitize for TTS — collapse \n\n+ to single space to avoid VoxCPM
        # rendering awkward pauses or literal punctuation between paragraphs.
        # Observed 2026-04-19: subconscious said "我是贾维斯。\n\n你好" which
        # sounded like "我是贾维斯 ...... 你好" with a stilted gap.
        import re as _re
        reply = _re.sub(r"\s*\n+\s*", " ", reply).strip()
        logger.info("RESPONDING · streaming TTS (%d chars)", len(reply))
        events.log("state", to="RESPONDING")
        events.log("tts_start", text=reply, chars=len(reply))
        t0 = time.monotonic()
        try:
            stream = backend_client.synth_stream(reply)
            await self.player.play_tts_stream(stream, sample_rate=cfg.TTS_SAMPLE_RATE)
            events.log("tts_done", elapsed_s=round(time.monotonic() - t0, 2))
        except Exception as e:
            logger.exception("TTS streaming failed")
            events.log("error", where="tts", msg=str(e))
            return State.ERROR
        return State.FOLLOW_UP_LISTEN

    # ---- ERROR: play error tone / short TTS, return IDLE --------------
    async def _run_error(self, message: str | None) -> State:
        logger.warning("ERROR · %s", message)
        # Play error wav (short beep) first for fast audio feedback
        try:
            await self.player.play_asset(cfg.ERROR_WAV)
        except Exception:
            pass
        # Optional spoken message via TTS (brief)
        if message and len(message) < 80:
            try:
                stream = backend_client.synth_stream(message)
                await self.player.play_tts_stream(stream, sample_rate=cfg.TTS_SAMPLE_RATE)
            except Exception:
                pass
        return State.IDLE

    # ---- FOLLOW_UP_LISTEN: stricter VAD + wake re-detection -----------
    async def _run_follow_up(self) -> tuple[State, bool]:
        """Returns (next_state, reset_session).

        Monitors the mic for:
          1. A wake-word trigger (score ≥ WAKE_THRESHOLD) → reset subconscious
             history and start a fresh conversation (user said "Hey Jarvis"
             again, wants a new topic, not a follow-up).
          2. Plain speech (strict VAD) → continue the current conversation
             as a follow-up turn (history kept).
          3. Nothing for FOLLOWUP_WINDOW_S → back to IDLE.
        """
        logger.info("FOLLOW_UP_LISTEN · %.0fs window (VAD %.2f, wake re-detect on)",
                    cfg.FOLLOWUP_WINDOW_S, cfg.VAD_FOLLOWUP_THRESHOLD)
        events.log("state", to="FOLLOW_UP_LISTEN")
        strict_vad = SileroVAD(threshold=cfg.VAD_FOLLOWUP_THRESHOLD)
        start = time.monotonic()
        wake_acc = np.zeros(0, dtype=np.int16)

        while time.monotonic() - start < cfg.FOLLOWUP_WINDOW_S:
            chunk = await self._next_chunk(timeout=0.25)
            if chunk is None:
                continue

            # Feed openWakeWord — if the user re-invokes we want a FRESH session.
            wake_acc = np.concatenate([wake_acc, chunk])
            while wake_acc.size >= WAKE_FRAME_SAMPLES:
                frame = wake_acc[:WAKE_FRAME_SAMPLES]
                wake_acc = wake_acc[WAKE_FRAME_SAMPLES:]
                try:
                    triggered, score = self.wake.predict(frame)
                except Exception:
                    triggered, score = False, 0.0
                if triggered:
                    logger.info(
                        "FOLLOW_UP_LISTEN · 🟢 WAKE re-triggered (score=%.3f) → new session",
                        score,
                    )
                    return State.LISTENING, True  # reset session flag

            # Meanwhile check strict VAD for follow-up content speech
            is_sp, prob = strict_vad.is_speech(chunk)
            if is_sp:
                logger.info("FOLLOW_UP_LISTEN · speech detected (p=%.2f) → LISTENING", prob)
                self.mic_q.put_nowait(chunk)
                return State.LISTENING, False

        logger.info("FOLLOW_UP_LISTEN · window expired → IDLE")
        return State.IDLE, False

    # ---- main loop ----------------------------------------------------
    async def run(self):
        loop = asyncio.get_running_loop()
        self.mic_q = asyncio.Queue(maxsize=200)
        self.mic = MicReader(loop, self.mic_q, self.player.mic_muted)
        self.mic.start()
        logger.info("satellite ready · state=%s", self.state.value)

        try:
            while True:
                if self.state == State.IDLE:
                    self.state = await self._run_idle()
                elif self.state == State.LISTENING:
                    # Distinguish first-wake listening from follow-up
                    next_state, audio = await self._run_listening(follow_up=False)
                    if next_state != State.PROCESSING:
                        self.state = next_state
                        continue
                    self.state, err_msg = await self._run_processing(audio)
                    if self.state == State.ERROR:
                        self.state = await self._run_error(err_msg)
                        continue
                    if self.state == State.RESPONDING:
                        self.state = await self._run_responding(err_msg)  # err_msg is the reply here
                elif self.state == State.FOLLOW_UP_LISTEN:
                    next_state, reset_session = await self._run_follow_up()
                    if next_state != State.LISTENING:
                        self.state = next_state
                        continue
                    if reset_session:
                        # Fresh "Hey Jarvis" heard — drop subconscious history
                        # so the new conversation starts from a clean slate.
                        logger.info("session reset · clearing subconscious history (%d msgs)",
                                    len(self.subconscious.history))
                        self.subconscious.history.clear()
                    # follow_up=False when session reset so we still play "在"
                    next_state2, audio = await self._run_listening(follow_up=not reset_session)
                    if next_state2 != State.PROCESSING:
                        self.state = next_state2
                        continue
                    self.state, err_or_reply = await self._run_processing(audio)
                    if self.state == State.ERROR:
                        self.state = await self._run_error(err_or_reply)
                    elif self.state == State.RESPONDING:
                        self.state = await self._run_responding(err_or_reply)
                else:
                    # PROCESSING/RESPONDING/ERROR are transient — should never be reached here
                    logger.warning("unexpected state in main dispatcher: %s", self.state)
                    self.state = State.IDLE
        finally:
            self.mic.stop()
            await backend_client.aclose()
            logger.info("satellite shut down")


# ============================================================
# Entrypoint
# ============================================================
def main():
    logging.basicConfig(
        level=cfg.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-devices", action="store_true")
    args = parser.parse_args()

    if args.dump_devices:
        for i, d in enumerate(sd.query_devices()):
            tag = []
            if d["max_input_channels"] >= 1:
                tag.append(f"in×{d['max_input_channels']}")
            if d["max_output_channels"] >= 1:
                tag.append(f"out×{d['max_output_channels']}")
            print(f"[{i:2d}] {d['name']:<50} ({', '.join(tag) or '-'})")
        return

    sat = Satellite()
    try:
        asyncio.run(sat.run())
    except KeyboardInterrupt:
        logger.info("interrupted")


if __name__ == "__main__":
    main()
