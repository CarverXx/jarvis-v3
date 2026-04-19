"""
Auto-trim the JARVIS (Paul Bettany) voice reference for VoxCPM2 cloning.

Problem: the source file (~/Downloads/YouTube/JARVIS ... .m4a, 24s) mixes BGM
plus multiple speakers. VoxCPM's zero-shot cloner learns the *spectral average*
of whatever is in the reference clip, so BGM + other voices poison the timbre
and cause the "音量忽大忽小、音色漂移" symptom Peter reported.

Strategy (BGM-robust — the trailer has music throughout, so silencedetect
alone can't carve clean dialogue segments):
  1. Decode source → mono 16 kHz f32.
  2. Sliding-window scan: for every START in [0, dur - WIN_S] step 0.25s,
     compute **frame-wise RMS stddev (dB)** over a WIN_S=6.0s window. Lowest
     stddev = most stable → least likely to contain music swells or speaker
     switches.
  3. Optional silencedetect pass removes windows whose START overlaps a
     long silence (>=1s) or that straddle the clip boundary.
  4. Extract with `ffmpeg -af "highpass=f=80,loudnorm=I=-20:LRA=7:TP=-3"`
     → 16 kHz mono s16le WAV, cleaner low end + consistent loudness.
  5. Write to `assets/jarvis_voice_ref.wav` (caller is expected to scp this
     to totodile and regen voxcpm_zai.wav).

Manual override: `--start 5.5 --end 11.2` skips auto-trim and cuts that range.

Usage:
    python scripts/trim_voice_ref.py                      # auto
    python scripts/trim_voice_ref.py --start 5.5 --end 11.2   # manual
    python scripts/trim_voice_ref.py --source ~/my.m4a --dry-run
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SOURCE = os.environ.get(
    "JARVIS_VOICE_SAMPLE_SOURCE",
    os.path.expanduser("~/Downloads/voice_sample.m4a"),
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "assets" / "jarvis_voice_ref.wav"

MIN_SEG_S = 4.0
MAX_SEG_S = 12.0
SILENCE_DB = -30
SILENCE_MIN_S = 0.3

# Sliding-window scan (primary picker — trailer has BGM throughout so
# silence-segment partition is unreliable).
WIN_S = 6.0          # length of candidate reference clip
HOP_S = 0.25         # step size when scanning
RMS_FRAME_MS = 50    # frame size for stddev


@dataclass
class Segment:
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"Segment({self.start:.2f}s–{self.end:.2f}s, dur={self.dur:.2f}s)"


def _run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False,
    )


def _probe_duration(path: str) -> float:
    r = _run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ], capture=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr}")
    return float(r.stdout.strip())


def _silencedetect(path: str) -> list[tuple[float, float]]:
    """Returns list of (silence_start, silence_end) tuples parsed from ffmpeg."""
    r = _run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", path,
        "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_S}",
        "-f", "null", "-",
    ], capture=True)
    silences: list[tuple[float, float]] = []
    cur_start: float | None = None
    for line in r.stderr.splitlines():
        if "silence_start:" in line:
            m = re.search(r"silence_start:\s*(-?[\d.]+)", line)
            if m:
                cur_start = float(m.group(1))
        elif "silence_end:" in line and cur_start is not None:
            m = re.search(r"silence_end:\s*(-?[\d.]+)", line)
            if m:
                silences.append((max(cur_start, 0.0), float(m.group(1))))
                cur_start = None
    return silences


def _invert_silence(silences: list[tuple[float, float]], total: float) -> list[Segment]:
    """Convert silence ranges into non-silence (speech-candidate) segments."""
    segs: list[Segment] = []
    cursor = 0.0
    for s, e in silences:
        if s - cursor > 0.1:
            segs.append(Segment(cursor, s))
        cursor = e
    if total - cursor > 0.1:
        segs.append(Segment(cursor, total))
    return segs


def _load_mono_16k(path: str) -> np.ndarray:
    """Decode source to mono 16 kHz f32 for RMS analysis."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp = tf.name
    try:
        r = _run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", path,
            "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", tmp,
        ], capture=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {r.stderr}")
        import wave
        with wave.open(tmp, "rb") as w:
            frames = w.readframes(w.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return audio
    finally:
        Path(tmp).unlink(missing_ok=True)


def _framewise_rms(audio: np.ndarray, sr: int = 16000, win_ms: int = 50) -> np.ndarray:
    win = int(sr * win_ms / 1000)
    n = len(audio) // win
    if n == 0:
        return np.array([0.0])
    trimmed = audio[: n * win].reshape(n, win)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def _score_segment(audio: np.ndarray, seg: Segment, sr: int = 16000) -> float:
    """Lower is more stable. Penalty = RMS stddev in dB within the segment.
    Segments containing BGM/music will have wide dynamic swings (high stddev);
    sustained dialogue will be flat."""
    s = int(seg.start * sr)
    e = int(seg.end * sr)
    chunk = audio[s:e]
    rms = _framewise_rms(chunk, sr)
    rms_db = 20 * np.log10(rms + 1e-9)
    return float(np.std(rms_db))


def pick_best_segment(source: str) -> Segment:
    """Sliding-window minimum-stddev picker — BGM-robust.
    We want the clip of length WIN_S whose short-term RMS is the flattest
    in dB. Music swells, speaker changes, and scene cuts all inflate
    stddev, so the minimum lands on sustained single-voice dialogue."""
    total = _probe_duration(source)
    audio = _load_mono_16k(source)
    sr = 16000

    # Optional: use silencedetect to reject windows whose start is within
    # a silence region (no point anchoring to dead air).
    silences = _silencedetect(source)

    def in_silence(t: float) -> bool:
        for s, e in silences:
            if s <= t <= e:
                return True
        return False

    frame = int(sr * RMS_FRAME_MS / 1000)
    win_frames = int(WIN_S * sr / frame)

    # Pre-compute per-frame RMS over the whole clip once.
    all_rms = _framewise_rms(audio, sr, RMS_FRAME_MS)
    all_rms_db = 20 * np.log10(all_rms + 1e-9)

    candidates: list[tuple[Segment, float, float]] = []  # (seg, stddev_db, mean_db)
    hop = HOP_S
    t = 0.0
    while t + WIN_S <= total + 1e-6:
        if not in_silence(t):
            start_idx = int(t * sr / frame)
            window = all_rms_db[start_idx : start_idx + win_frames]
            if len(window) >= win_frames // 2:
                stddev = float(np.std(window))
                mean = float(np.mean(window))
                # Reject windows that are mostly silence (mean too low).
                if mean > -55.0:
                    candidates.append((Segment(t, t + WIN_S), stddev, mean))
        t += hop

    if not candidates:
        print(f"[pick] WARN no candidate windows — falling back to [0, {min(total, WIN_S):.2f}]", file=sys.stderr)
        return Segment(0.0, min(total, WIN_S))

    candidates.sort(key=lambda x: x[1])
    print(f"[pick] scanned {len(candidates)} windows × {WIN_S}s; top 5 by stability:", file=sys.stderr)
    for seg, std, mean in candidates[:5]:
        print(f"  · {seg}  stddev={std:.2f} dB  mean={mean:.1f} dBFS", file=sys.stderr)

    best = candidates[0][0]
    print(f"[pick] → selected {best}", file=sys.stderr)
    return best


def extract(source: str, start: float, end: float, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # backup existing
    if out_path.exists():
        bak = out_path.with_suffix(out_path.suffix + ".prev")
        shutil.copy2(out_path, bak)
        print(f"[extract] backed up existing → {bak}", file=sys.stderr)

    af = "highpass=f=80,loudnorm=I=-20:LRA=7:TP=-3"
    r = _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", source,
        "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-af", af,
        str(out_path),
    ], capture=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg extract failed: {r.stderr}")
    print(f"[extract] wrote {out_path} ({start:.2f}s–{end:.2f}s, af='{af}')", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=DEFAULT_SOURCE)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--start", type=float, help="manual start (s), skips auto-pick")
    p.add_argument("--end", type=float, help="manual end (s), skips auto-pick")
    p.add_argument("--dry-run", action="store_true", help="pick segment but don't write")
    args = p.parse_args()

    if not os.path.exists(args.source):
        print(f"source not found: {args.source}", file=sys.stderr)
        return 2

    if args.start is not None and args.end is not None:
        seg = Segment(args.start, args.end)
        print(f"[manual] using {seg}", file=sys.stderr)
    elif args.start is None and args.end is None:
        seg = pick_best_segment(args.source)
    else:
        print("--start and --end must be given together", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"[dry-run] would extract {seg} → {args.out}")
        return 0

    extract(args.source, seg.start, seg.end, Path(args.out))
    print(f"OK → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
