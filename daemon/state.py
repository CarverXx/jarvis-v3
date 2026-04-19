"""
JARVIS v3 state machine — names from Home Assistant's AssistSatelliteState
(homeassistant/core `components/assist_satellite/entity.py`), extended with
FOLLOW_UP_LISTEN for OVOS-style hybrid multi-turn, and ERROR for recovery.
"""
from __future__ import annotations

from enum import Enum


class State(str, Enum):
    IDLE = "idle"                          # listening for wake word only
    LISTENING = "listening"                # post-wake, capturing command audio
    PROCESSING = "processing"              # ASR + Hermes call in flight
    RESPONDING = "responding"              # TTS playing, mic muted
    FOLLOW_UP_LISTEN = "follow_up_listen"  # bot finished, waiting for re-query
    ERROR = "error"                        # transient fault; play error tone
