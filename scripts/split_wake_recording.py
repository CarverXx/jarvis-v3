"""
Split a single recording (m4a/mp3/wav) containing multiple "Hey Jarvis"
utterances into individual 2-second clips suitable for training an
openWakeWord custom verifier model.

Strategy (BGM-free, clean-environment recording):
  1. Decode source → mono 16 kHz float32 via ffmpeg.
  2. Compute frame-wise RMS (25ms hop, 50ms window).
  3. Mark frames where RMS > threshold as "voiced", merge adjacent voiced
     frames into contiguous SPANS (with up to `merge_gap_ms` of silence
     merged inside a span to keep a single "Hey Jarvis" intact).
  4. Keep spans whose duration ∈ [min_span_s, max_span_s] — filters out
     brief coughs/noises and runs where Peter said the word too quickly.
  5. For each span, extract a `clip_s`-second window centered on the span's
     RMS centroid. Saves as 16-bit signed mono 16kHz WAV to out_dir.

Output filenames: 00.wav, 01.wav, 02.wav, … (zero-padded 2 digits).

Usage:
    python scripts/split_wake_recording.py \\
        --input ~/Downloads/hey_jarvis_batch.m4a \\
        --out-dir assets/wake_samples/positive \\
        [--n 15] [--clip-s 2.0] [--rms-thresh -35] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SR = 16000          # openWakeWord requires 16kHz mono
DEFAULT_CLIP_S = 2.0
DEFAULT_N = 15
DEFAULT_RMS_THRESH_DB = -35.0    # above this = voiced
DEFAULT_MIN_SPAN_S = 0.4
DEFAULT_MAX_SPAN_S = 2.0
DEFAULT_MERGE_GAP_MS = 150       # merge sub-spans within this much silence


@dataclass
class Span:
    start_s: float
    end_s: float
    peak_rms_db: float
    centroid_s: float

    @property
    def dur(self) -> float:
        return self.end_s - self.start_s

    def __repr__(self) -> str:
        return (f"Span({self.start_s:.2f}-{self.end_s:.2f}s dur={self.dur:.2f}s "
                f"peak={self.peak_rms_db:.1f}dB centroid={self.centroid_s:.2f}s)")


def decode_to_mono16k(src: str) -> np.ndarray:
    """ffmpeg-decode any audio file to 16kHz mono float32 ndarray."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp = tf.name
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", src, "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", tmp],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {r.stderr}")
        with wave.open(tmp, "rb") as w:
            frames = w.readframes(w.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    finally:
        Path(tmp).unlink(missing_ok=True)


def frame_rms(audio: np.ndarray, hop_ms: int = 25, win_ms: int = 50) -> tuple[np.ndarray, int, int]:
    """Returns (rms_per_frame, hop_samples, win_samples)."""
    hop = int(SR * hop_ms / 1000)
    win = int(SR * win_ms / 1000)
    n_frames = max(0, (len(audio) - win) // hop + 1)
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        frame = audio[s:s + win]
        rms[i] = float(np.sqrt(np.mean(frame ** 2)))
    return rms, hop, win


def find_spans(
    rms: np.ndarray,
    hop: int,
    rms_thresh_db: float,
    min_span_s: float,
    max_span_s: float,
    merge_gap_ms: int,
) -> list[Span]:
    """Identify voiced spans where rms > threshold, merging gaps < merge_gap_ms."""
    rms_db = 20 * np.log10(rms + 1e-9)
    voiced = rms_db > rms_thresh_db

    hop_s = hop / SR
    merge_gap_frames = int(merge_gap_ms / 1000 / hop_s)
    min_span_frames = int(min_span_s / hop_s)
    max_span_frames = int(max_span_s / hop_s)

    spans: list[Span] = []
    i = 0
    n = len(voiced)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        # Start of a span
        span_start = i
        # Walk forward, allowing up to merge_gap_frames of silence internally
        j = i
        while j < n:
            if voiced[j]:
                j += 1
                continue
            # Look ahead for next voiced within merge_gap
            k = j
            while k < n and k - j < merge_gap_frames:
                if voiced[k]:
                    break
                k += 1
            if k < n and voiced[k]:
                # Gap was short enough; absorb it
                j = k + 1
            else:
                break
        span_end = j
        # Record span if within size constraints
        dur_frames = span_end - span_start
        if min_span_frames <= dur_frames <= max_span_frames:
            s_rms = rms[span_start:span_end]
            # Centroid weighted by energy (rms^2)
            weights = s_rms ** 2
            if weights.sum() > 0:
                rel_centroid_frame = float((weights * np.arange(len(weights))).sum() / weights.sum())
            else:
                rel_centroid_frame = len(weights) / 2.0
            spans.append(Span(
                start_s=span_start * hop_s,
                end_s=span_end * hop_s,
                peak_rms_db=float(20 * np.log10(s_rms.max() + 1e-9)),
                centroid_s=(span_start + rel_centroid_frame) * hop_s,
            ))
        i = span_end + 1
    return spans


def clip_span(audio: np.ndarray, span: Span, clip_s: float) -> np.ndarray:
    """Extract a clip_s-second window centered on the span's centroid,
    clamped to the audio bounds. Zero-pads if the centroid is near edges."""
    total_s = len(audio) / SR
    half = clip_s / 2
    center = span.centroid_s
    start_s = max(0.0, center - half)
    end_s = min(total_s, center + half)
    # If we hit an edge, shift the window to preserve clip_s length when possible
    if end_s - start_s < clip_s:
        if start_s == 0.0:
            end_s = min(total_s, clip_s)
        elif end_s == total_s:
            start_s = max(0.0, total_s - clip_s)
    start = int(start_s * SR)
    end = int(end_s * SR)
    clip = audio[start:end]
    # Zero-pad to exact clip_s length
    target_len = int(clip_s * SR)
    if len(clip) < target_len:
        clip = np.pad(clip, (0, target_len - len(clip)))
    return clip


def save_wav_int16(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="path to recording (m4a/mp3/wav)")
    p.add_argument("--out-dir", required=True, help="output directory (cleared then written)")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="max number of clips to keep (top-N by peak RMS)")
    p.add_argument("--clip-s", type=float, default=DEFAULT_CLIP_S)
    p.add_argument("--rms-thresh", type=float, default=DEFAULT_RMS_THRESH_DB, help="dBFS threshold for voiced")
    p.add_argument("--min-span-s", type=float, default=DEFAULT_MIN_SPAN_S)
    p.add_argument("--max-span-s", type=float, default=DEFAULT_MAX_SPAN_S)
    p.add_argument("--merge-gap-ms", type=int, default=DEFAULT_MERGE_GAP_MS)
    p.add_argument("--dry-run", action="store_true", help="report spans but don't write files")
    p.add_argument("--clear-out", action="store_true", help="rm -rf out_dir before writing (default: safe merge)")
    args = p.parse_args()

    src = os.path.expanduser(args.input)
    if not os.path.exists(src):
        print(f"input not found: {src}", file=sys.stderr)
        return 2

    print(f"[decode] {src} → mono 16kHz", file=sys.stderr)
    audio = decode_to_mono16k(src)
    total = len(audio) / SR
    print(f"[decode] duration={total:.2f}s samples={len(audio)}", file=sys.stderr)

    rms, hop, win = frame_rms(audio)
    spans = find_spans(rms, hop, args.rms_thresh, args.min_span_s, args.max_span_s, args.merge_gap_ms)
    print(f"[spans] {len(spans)} candidates (rms_thresh={args.rms_thresh}dB, merge_gap={args.merge_gap_ms}ms)", file=sys.stderr)
    for s in spans:
        print(f"  · {s}", file=sys.stderr)

    # Rank by peak RMS and take top-N
    spans.sort(key=lambda s: s.peak_rms_db, reverse=True)
    kept = spans[:args.n]
    # Re-sort by time for stable output numbering
    kept.sort(key=lambda s: s.start_s)
    print(f"[kept] top {len(kept)} spans by peak-RMS:", file=sys.stderr)
    for i, s in enumerate(kept):
        print(f"  [{i:02d}] {s}", file=sys.stderr)

    if args.dry_run:
        print("[dry-run] skipping write", file=sys.stderr)
        return 0

    out_dir = Path(os.path.expanduser(args.out_dir))
    if args.clear_out and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, span in enumerate(kept):
        clip = clip_span(audio, span, args.clip_s)
        out_path = out_dir / f"{i:02d}.wav"
        save_wav_int16(out_path, clip)
        print(f"[write] {out_path} ({args.clip_s:.1f}s @ {SR}Hz)", file=sys.stderr)

    print(f"OK → {len(kept)} clips in {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
