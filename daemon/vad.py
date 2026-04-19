"""
Silero VAD wrapper.

Replaces WebRTC VAD for the RECORD state.  Silero is more accurate
(fewer false positives on music/noise, better multi-lingual support
for Chinese + English speech detection) at the cost of ~1 ms per chunk
on CPU.

Frame contract: **512 int16 mono samples per call @ 16 kHz**
(32 ms windows, required by Silero VAD v5+).
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)

CHUNK_SAMPLES = 512  # 32 ms @ 16 kHz (required by silero-vad)


class SileroVAD:
    """Lazy-load Silero VAD and expose per-chunk is_speech() API."""

    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000) -> None:
        from silero_vad import load_silero_vad

        self._threshold = threshold
        self._sr = sampling_rate
        self._model = load_silero_vad(onnx=True)
        logger.info("SileroVAD loaded (threshold=%.2f, sr=%d)", threshold, sampling_rate)

    def is_speech(self, chunk_int16: np.ndarray) -> tuple[bool, float]:
        """Feed one 512-sample int16 mono chunk. Returns (is_speech, probability)."""
        import torch

        if chunk_int16.dtype != np.int16:
            chunk_int16 = chunk_int16.astype(np.int16)
        if chunk_int16.ndim > 1:
            chunk_int16 = chunk_int16.flatten()
        # Pad / trim to CHUNK_SAMPLES (Silero strict requirement)
        if len(chunk_int16) < CHUNK_SAMPLES:
            chunk_int16 = np.pad(chunk_int16, (0, CHUNK_SAMPLES - len(chunk_int16)))
        elif len(chunk_int16) > CHUNK_SAMPLES:
            chunk_int16 = chunk_int16[:CHUNK_SAMPLES]

        # Silero expects float32 tensor in range [-1, 1]
        chunk_f32 = chunk_int16.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(chunk_f32)
        prob = float(self._model(tensor, self._sr).item())
        return prob >= self._threshold, prob

    def reset_states(self) -> None:
        """Reset the RNN hidden state between utterances."""
        # Silero OnnxWrapper exposes reset_states()
        try:
            self._model.reset_states()
        except AttributeError:
            pass
