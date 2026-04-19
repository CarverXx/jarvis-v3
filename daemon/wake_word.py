"""
Wake word detector wrapper around openWakeWord.

Uses the built-in "hey_jarvis_v0.1" ONNX model that ships with the
openwakeword package. CPU inference (<5% on modern CPUs), no GPU needed.

Frame contract: **1280 int16 mono samples per call @ 16 kHz**
(80 ms windows, same as openWakeWord's internal streaming protocol).
The daemon feeds one 1280-sample frame per iteration and checks the
returned score against the configured threshold.

Custom verifier model (optional): if `assets/hey_jarvis_verifier.pkl` exists,
it's loaded automatically. openWakeWord then multiplies the base wake score
by the verifier's cosine-similarity-based probability that the speaker is
Peter, filtering out other voices / TV / echo. Pkl is trained offline via
`scripts/enroll_wake_word` + `train_custom_verifier()` — see Phase 6 plan.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz (required by openwakeword)

# Default locations relative to the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VERIFIER_PATH = _REPO_ROOT / "assets" / "hey_jarvis_verifier.pkl"


# Built-in "hey_jarvis" ONNX model path inside the openwakeword pkg.
# Resolved lazily so we only import openwakeword if the detector is actually used.
def _default_hey_jarvis_path() -> str:
    import openwakeword
    pkg_dir = Path(openwakeword.__file__).parent
    p = pkg_dir / "resources" / "models" / "hey_jarvis_v0.1.onnx"
    if not p.exists():
        raise FileNotFoundError(
            f"hey_jarvis model not found at {p} — reinstall openwakeword"
        )
    return str(p)


class WakeWordDetector:
    """Lazy-load openWakeWord "hey_jarvis" and expose a simple predict() API."""

    def __init__(
        self,
        model_path: str | None = None,
        threshold: float = 0.5,
        vad_threshold: float = 0.0,  # 0 = disable internal VAD pre-filter
        verifier_path: str | None = None,
        verifier_threshold: float = 0.1,
    ) -> None:
        from openwakeword import Model

        self._threshold = threshold
        self._model_path = model_path or _default_hey_jarvis_path()

        # Auto-detect verifier pkl if not explicitly passed.
        resolved_verifier = verifier_path or os.environ.get("WAKE_VERIFIER_PATH")
        if not resolved_verifier and DEFAULT_VERIFIER_PATH.exists():
            resolved_verifier = str(DEFAULT_VERIFIER_PATH)

        verifier_kwargs: dict = {}
        if resolved_verifier and os.path.exists(resolved_verifier):
            # model key for verifier dict must match the onnx filename stem.
            model_stem = Path(self._model_path).stem  # "hey_jarvis_v0.1"
            verifier_kwargs = {
                "custom_verifier_models": {model_stem: resolved_verifier},
                "custom_verifier_threshold": verifier_threshold,
            }

        self._model = Model(
            wakeword_model_paths=[self._model_path],
            vad_threshold=vad_threshold,
            **verifier_kwargs,
        )
        self._key = next(iter(self._model.models))  # e.g. "hey_jarvis_v0.1"

        verifier_status = (
            f"verifier={Path(resolved_verifier).name} (gate≥{verifier_threshold})"
            if resolved_verifier and os.path.exists(resolved_verifier)
            else "verifier=off"
        )
        logger.info(
            "WakeWordDetector loaded %s (threshold=%.2f, %s)",
            self._key, self._threshold, verifier_status,
        )

    @property
    def key(self) -> str:
        return self._key

    @property
    def threshold(self) -> float:
        return self._threshold

    def predict(self, frame_int16: np.ndarray) -> tuple[bool, float]:
        """Feed one 1280-sample int16 mono frame (80 ms @ 16 kHz).

        Returns (triggered, score) where triggered = score >= threshold.
        """
        if frame_int16.dtype != np.int16:
            frame_int16 = frame_int16.astype(np.int16)
        if frame_int16.ndim > 1:
            frame_int16 = frame_int16.flatten()
        if len(frame_int16) != FRAME_SAMPLES:
            # openwakeword internally manages state — non-ideal frame size
            # still works but we warn once.
            logger.debug(
                "wake predict got %d samples, expected %d", len(frame_int16), FRAME_SAMPLES
            )
        scores = self._model.predict(frame_int16)
        score = float(scores.get(self._key, 0.0))
        return score >= self._threshold, score

    def reset(self) -> None:
        """Reset internal streaming state (clear history buffer)."""
        # openWakeWord Model has no public reset; re-instantiate via:
        # self._model = Model(wakeword_model_paths=[self._model_path], vad_threshold=0)
        # For now we don't need reset — the model auto-decays its state.
        pass
