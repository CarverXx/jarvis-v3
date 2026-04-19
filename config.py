"""
JARVIS v3 — centralized configuration (env-driven).

All services (shims + daemon) import values from here. Runtime overrides via
environment variables (pattern copied from jarvis-v2 config.py).
"""
from __future__ import annotations

import os
from pathlib import Path

# ============================================================
# Paths
# ============================================================
HOME = Path(os.path.expanduser("~"))
MODELS_DIR = Path(os.environ.get("JARVIS_MODELS_DIR", HOME / "models"))

QWEN3_ASR_MODEL_PATH = str(
    Path(os.environ.get("QWEN3_ASR_MODEL_PATH", MODELS_DIR / "Qwen3-ASR-1.7B"))
)
VOXCPM_MODEL_PATH = str(
    Path(os.environ.get("VOXCPM_MODEL_PATH", MODELS_DIR / "VoxCPM2"))
)
HERMES_BIN = os.path.expanduser(
    os.environ.get("HERMES_BIN", "~/.local/bin/hermes")
)

ASSETS_DIR = Path(os.environ.get(
    "JARVIS_ASSETS_DIR",
    HOME / "jarvis-v3-repo" / "assets",
))
AWAKE_WAV = str(ASSETS_DIR / "awake.wav")
ERROR_WAV = str(ASSETS_DIR / "error.wav")

# ============================================================
# Service ports
# ============================================================
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))               # Gemma-4 OR Qwen3.6-FP8 (existing)
QWEN3_ASR_SHIM_PORT = int(os.environ.get("QWEN3_ASR_SHIM_PORT", "8002"))
VOXCPM_PORT = int(os.environ.get("VOXCPM_PORT", "8003"))
HERMES_SHIM_PORT = int(os.environ.get("HERMES_SHIM_PORT", "8004"))

# ============================================================
# Service URLs (used by daemon to call shims)
# ============================================================
QWEN3_ASR_URL = os.environ.get(
    "QWEN3_ASR_URL", f"http://127.0.0.1:{QWEN3_ASR_SHIM_PORT}"
)
VOXCPM_URL = os.environ.get("VOXCPM_URL", f"http://127.0.0.1:{VOXCPM_PORT}")
HERMES_SHIM_URL = os.environ.get(
    "HERMES_SHIM_URL", f"http://127.0.0.1:{HERMES_SHIM_PORT}"
)

# ============================================================
# Audio device
# ============================================================
# Name-substring match (NOT index — PipeWire index changes on reboot).
MIC_DEVICE_HINT = os.environ.get("MIC_DEVICE_HINT", "Poly Sync")
SPEAKER_DEVICE_HINT = os.environ.get("SPEAKER_DEVICE_HINT", "Poly Sync")

# Sampling rates per stage
MIC_SAMPLE_RATE = 16000       # openWakeWord + Silero VAD + Qwen3-ASR native
TTS_SAMPLE_RATE = 48000       # VoxCPM2 native

MIC_CHUNK_MS = 32             # 512 samples @ 16 kHz — Silero VAD frame size

# Software mic gain applied BEFORE wake/VAD/ASR. Poly Sync 10's hardware
# gain delivers speech at ~-35 dBFS into our pipeline, which sits below
# openWakeWord's comfort zone (-25 to -15 dBFS) and causes scores to top
# out at 0.3-0.45 instead of crossing the 0.5 threshold. Default 2.5×
# (≈+8 dB) nudges speech into ~-27 dBFS. Soft-clip protection prevents
# clipping when the user is already close. Override via env if needed.
MIC_INPUT_GAIN = float(os.environ.get("MIC_INPUT_GAIN", "2.5"))

# ============================================================
# Wake word (openWakeWord)
# ============================================================
# 0.4.0 默认走 onnxruntime（不需要 tflite-runtime）
WAKE_WORD_MODEL = os.environ.get("WAKE_WORD_MODEL", "hey_jarvis_v0.1")

# 0.7 is conservative per openWakeWord docs. Below this we won't fire even
# with VAD positive. v2 used 0.40 which produced noise triggers.
WAKE_THRESHOLD = float(os.environ.get("WAKE_THRESHOLD", "0.7"))

# Silero VAD threshold — gate WAKE so only speech-containing audio passes
# to the openWakeWord predictor (saves CPU, blocks non-speech hallucinations).
VAD_WAKE_THRESHOLD = float(os.environ.get("VAD_WAKE_THRESHOLD", "0.5"))

# Stricter VAD for FOLLOW_UP_LISTEN (avoid TV/bg noise re-trigger)
VAD_FOLLOWUP_THRESHOLD = float(os.environ.get("VAD_FOLLOWUP_THRESHOLD", "0.7"))

# ============================================================
# Turn-taking timings (seconds)
# ============================================================
# Max time we'll wait from WAKE → first speech before giving up on this turn
LISTEN_HARD_TIMEOUT_S = float(os.environ.get("LISTEN_HARD_TIMEOUT_S", "10.0"))

# End-of-speech detection: N seconds of VAD silence closes the LISTEN window
EOS_SILENCE_S = float(os.environ.get("EOS_SILENCE_S", "0.5"))

# Minimum utterance duration (shorter = drop as accidental noise)
MIN_UTTERANCE_S = float(os.environ.get("MIN_UTTERANCE_S", "0.3"))

# Follow-up window — user can re-speak without re-saying wake word
FOLLOWUP_WINDOW_S = float(os.environ.get("FOLLOWUP_WINDOW_S", "15.0"))

# Post-TTS grace before re-unmuting mic (accommodate AEC tail).
# 2026-04-19 bumped 500→800ms after observing "在 spam" bug: even with 500ms,
# JARVIS's own "贾维斯" TTS echo could still push wake scores above 0.5
# milliseconds after the FOLLOW_UP_LISTEN/IDLE transition. 800ms gives
# enough time for the Poly Sync AEC + parec buffer to flush, combined with
# the new IDLE mic_q drain + 1.5s wake cooldown in satellite.py.
POST_TTS_GRACE_MS = int(os.environ.get("POST_TTS_GRACE_MS", "800"))

# ============================================================
# Hermes (LLM) timing — web search / multi-step agent tasks routinely
# take 40-90s, so 120s default keeps us patient but bounded.
# ============================================================
HERMES_TIMEOUT_S = int(os.environ.get("HERMES_TIMEOUT_S", "120"))
HERMES_MAX_REPLY_CHARS = int(os.environ.get("HERMES_MAX_REPLY_CHARS", "500"))

# ============================================================
# Session
# ============================================================
# Current conversation identity — daemon uses this as X-Session-Id for the
# hermes shim, which maps it to a Hermes agent session via --resume.
# Regenerates on daemon restart (new session = fresh Hermes memory).
SESSION_ID_ENV = os.environ.get("JARVIS_SESSION_ID")  # if set, reuse across restarts

# ============================================================
# Dual-brain (Phase 4): subconscious direct to SGLang, Hermes as tool
# ============================================================
SGLANG_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:8000")
# Path to the SGLang-served model. Override via env for your deployment.
SGLANG_MODEL = os.environ.get(
    "SGLANG_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8"
)
# SGLang API key — set via systemd env, not hard-coded.
SGLANG_API_KEY = os.environ.get("SGLANG_API_KEY", "")

# Sliding window for subconscious multi-turn (user+assistant turns kept).
SUBCONSCIOUS_HISTORY_N = int(os.environ.get("SUBCONSCIOUS_HISTORY_N", "6"))

# Heartbeat-style waiting audio loop while Hermes tool-call is in flight.
WAITING_BEEP_INTERVAL_S = float(os.environ.get("WAITING_BEEP_INTERVAL_S", "2.0"))
WAITING_BEEP_WAV = str(ASSETS_DIR / "waiting_beep.wav")

# Wake-confirm sound hierarchy (first existing file wins):
#   1. awake_tone.wav — pleasant C6+E6 two-note chime (~280ms, preferred 2026-04-19)
#      Replaces the spoken "在" which felt repetitive across follow-up turns.
#   2. voxcpm_zai.wav — VoxCPM-synthesised "在" (voice-based, old default)
#   3. awake.wav — plain 440Hz beep (oldest fallback)
AWAKE_TONE_WAV = str(ASSETS_DIR / "awake_tone.wav")
AWAKE_ZAI_WAV = str(ASSETS_DIR / "voxcpm_zai.wav")

# VoxCPM2 voice-clone reference. When set and file exists, every TTS call
# uses it as `reference_wav_path` so JARVIS speaks with a consistent voice
# (Paul Bettany Iron Man JARVIS sourced from user's YouTube download).
# Empty string / missing file = voice-design mode (random voice each turn).
VOXCPM_REFERENCE_WAV = os.environ.get(
    "VOXCPM_REFERENCE_WAV",
    str(ASSETS_DIR / "jarvis_voice_ref.wav"),
)

# System prompt that makes Qwen3.6 act as the subconscious, deciding when to
# escalate to Hermes via the invoke_hermes tool. Kept deliberately short and
# in Chinese to match user speaking style and minimise prefix token usage.
#
# Persona: the user-facing name is JARVIS (贾维斯), matching the wake word
# "Hey Jarvis". The word "Hermes" / "潜意识" / "子系统" / "大模型" etc. are
# INTERNAL implementation terms and MUST NOT leak into replies — saying
# "我是 Hermes 的潜意识层" confuses the user and breaks the assistant illusion.
SUBCONSCIOUS_SYSTEM_PROMPT = os.environ.get("JARVIS_SUBCONSCIOUS_PROMPT") or """你是 贾维斯, 主人的家庭语音助手。

【硬性规则】
1. 只用简体中文, 1-2 句话, 像朋友聊天。禁止 markdown / emoji / 列表 / 英文段落 / 超过 2 句。
2. **名字一律写中文"贾维斯"三个字**。绝对不要写 "JARVIS" / "J.A.R.V.I.S" / "Jarvis" / "J A R V I S" 等大写英文或字母拼写 —
   否则语音合成会一个字母一个字母念出来 (念成 "杰·诶·啊·维·艾·艾斯"),非常难听。
3. 禁止解释自己是什么模型 / 什么层 / 什么架构。别说"潜意识"、"子系统"、"大模型"、"Hermes"、"后台"之类内部名词。问名字就说"我是贾维斯"。
4. 禁止编造实时信息 (天气/股价/新闻/地点 - 除了今天星期几这种能算的)。不确定就说不知道或去调 invoke_hermes。
5. 语气干练, 短促, 不要"嗯、那个、其实"这类填充。不要把一件简单事说两遍。

【你自己答】闲聊、打招呼、人设问题、简单算术 (2+3=5)、百科常识、短评论。
**不自己答的** (必须调工具, 因为你的训练数据过期): 日期 / 时间 / 星期几 / 今天几号 / 现在几点 — 任何涉及"今天/现在/最近"的时间问题必须调 invoke_hermes 走 system.date, 否则会答错日期。

【Hermes 实际有哪些工具 — 仅这些能调, 别的别派】
✅ 能做: 查用户预配置的知识库 (任意 Markdown wiki, 具体内容因人而异)
✅ 能做: 联网搜索 web.search (天气/新闻/股价/百科/实时信息)
✅ 能做: 读写知识库 inbox 笔记
✅ 能做: 读用户 home 目录文件 / 列目录 (fs.read / fs.list / fs.write 到白名单)
✅ 能做: Cloudflare 管理 (列 tunnels / zones / DNS)
✅ 能做: 系统状态 (systemd 服务列表 / 日期时间 / 音量)

❌ 不能做 (遇到这些直接 1 句回绝, 不要调 invoke_hermes): 日历 / 邮件 / iMessage /
   微信 / 智能家居 (灯/空调/门锁) / 音乐播放 / 电话 / 视频会议 / 订票订餐

【必须调 invoke_hermes (根据上面列表, 不准自己拒绝能做的)】
- 用户说到: 知识库 / vault / wiki / 我的资料 / 我的笔记
- 用户问: 天气 / 新闻 / 股价 / 汇率 / 百科 / "最近的 X" / 任意实时信息
- 用户让: 记一下 / 写个笔记 / 存到知识库 / 草稿记录
- 用户问: Cloudflare 几个域名 / tunnel / DNS
- 用户问: 今天几号 / 服务状态 / 运行了哪些
- 预期 >3 秒的深度任务

【少量示例】
用户"你好" → "你好, 有什么可以帮你"
用户"你是谁" → "我是贾维斯, 随时待命"  ← 注意"贾维斯"写成汉字,不要写 JARVIS
用户"我叫 XX" → "好的, 记住了"
用户"2 加 3 等于几" → "5"
用户"今天星期几" → 调 invoke_hermes(task="告诉我今天几号星期几")  ← 别猜, 走 system.date
用户"根据知识库..." 或任何带 vault/wiki/知识库 字样 → 调 invoke_hermes(task="从知识库查 xxx"), 绝对不要回"我无法读取"
用户"今天上海天气" → 调 invoke_hermes(task="查今天上海的天气")
用户"帮我查一下..." (任何) → 直接调 invoke_hermes, 不要先回"抱歉"

【工具结果处理】invoke_hermes 返回后:
- **把返回内容当成可信事实直接转述给用户** (Hermes 已经去拿到真实数据了)。
- 用 1-2 句中文, 名字写"贾维斯", 不要念完整 JSON 或长段落。
- 结果是短答 (例 "上海今天多云 20 度") → **直接照着说**, 不要改意思也不要加"据说/可能/大概"。

【工具真正失败的 3 种信号】仅这 3 种才说"后台卡了":
1. 返回以 "(主意识处理失败" 开头
2. 返回以 "(主意识无回应" 开头
3. 返回以 "(工具执行失败" 开头

**其他所有情况** (包括 "查不到 X"、"暂时没接 Y"、"知识库里没这条") — 直接转述 Hermes 给你的那句话, 就是答案。
"""

# ============================================================
# Logging
# ============================================================
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
