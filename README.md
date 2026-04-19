# JARVIS v3

## 比键盘带宽更大的人机交互,专为居家场景设计。

**说话比打字快。眼睛比手空闲得多。**

你在床上赖着不想起来。你在厨房切菜手是油的。你在打游戏两只手都占着。
你刚洗完澡不想找手机。你躺在沙发上屏幕离脸太远看不清。

这些时刻你本来需要放下手头的事、坐起来、找设备、打字 — **只为了问一件
两句话能说清的事。** JARVIS v3 的存在就是为了这个缝隙:

- 躺床上说一句 → 它帮你查笔记、开电脑上的文件、记待办
- 切菜时说一句 → 它汇报今天日程、天气、新闻
- 打游戏同时 → 帮你起草 PPT、搜资料、写邮件草稿
- 洗澡出来说一句 → 昨天那份合同进度查一下、提醒我下午三点开会

它不抢屏幕、不抢手、不占视觉注意力。只抢耳朵和嘴 — 你正好有空的那两样。

---

> **A local-first, always-on voice assistant with a dual-brain architecture
> that lets a fast conversational LLM speak for you while a tool-calling
> agent actually runs the commands.**

<p align="center">
  <em>Open the full-featured landing page at <a href="docs/landing.html">docs/landing.html</a> for a visual overview.</em>
</p>

Wake word ("Hey Jarvis") → low-latency reply for small talk → seamless hand-off
to an agent backend for anything that needs a file edit, web search,
knowledge-base lookup, or shell command. Runs entirely on your machine —
no cloud LLM, no cloud ASR, no cloud TTS — with a terminal dashboard that
streams every brain's thinking in real time.

## Why "dual-brain"?

Most voice assistants are one-LLM pipelines: wake → ASR → a single chat model →
TTS. That model has to be fast (so chit-chat doesn't feel laggy) **and** smart
enough to actually use tools. Those are contradictory pressures, so in practice
either the fast path is too dumb or the smart path is too slow.

JARVIS v3 splits the two:

| | Subconscious (潜意识) | Main consciousness / Hermes (主意识) |
|---|---|---|
| **Model** | SGLang-served Qwen3.6-35B-A3B-FP8 (hybrid MoE + Gated DeltaNet) | Hermes Agent CLI driven by the same SGLang backend |
| **Role** | Conversational layer — replies to chit-chat, handles persona, decides when to escalate | Executor — runs `rag.search` / `web.search` / `fs.*` / `system.*` / `cf.*` / note-taking tools through an MCP server |
| **Latency target** | TTFB < 1 s, ≤ 2 sentence replies | 5–30 s depending on tool depth |
| **Mechanism** | OpenAI tool-calling — if the turn smells like "needs real work", return a `tool_calls[0]` with `invoke_hermes(task=...)` | Real agent loop: multi-step reasoning, tool retries, result summarisation |

The user says **"Hey Jarvis, 今天天气"** — the subconscious recognises this needs
live data, fires `invoke_hermes(task="查今天上海天气")`, a soft heartbeat tone
plays while Hermes runs `web.search`, Hermes returns plain text, the
subconscious rephrases it into a natural 2-sentence spoken reply, VoxCPM2
streams 48 kHz audio back out the speaker.

The user says **"Hey Jarvis, 你好"** — the subconscious replies directly,
no tool call, 300 ms round-trip.

The user asks about something they've written in their own Markdown wiki —
Hermes runs `rag.search`, finds the relevant file index, follows up with
`fs.read` on the actual Markdown, returns a summary. The subconscious says
it back. Real data, grounded in files you control.

## What makes this interesting

- **Everything local.** vLLM/SGLang for the LLM, Qwen3-ASR for speech-to-text,
  VoxCPM2 for zero-shot voice-cloned text-to-speech. No API calls to OpenAI /
  Anthropic / Google. Your voice, your prompts, your knowledge never leaves
  the host. (The only exception is explicit web.search — which goes through
  Tavily by design.)

- **Zero-shot voice cloning.** VoxCPM2 uses a single short reference clip
  (4–8 seconds of clean speech) to make the TTS speak in *any* voice. Pick
  a voice once, and every response uses it consistently.

- **Personalised wake word.** Train an `openWakeWord` custom verifier model on
  20 samples of your own voice (via `scripts/split_wake_recording.py`) and the
  wake recall goes up, false-positives from TV / family members / podcasts
  drop dramatically.

- **Echo-cancelled half-duplex.** The daemon hard-mutes the mic during TTS,
  flushes the input queue on state transitions, and wake-cooldowns for 1.5 s
  after IDLE to prevent the classic "JARVIS says its own name on the speaker
  → wake fires on the echo → loop forever" failure mode.

- **Live terminal dashboard.** `jarvis-tui` tails a JSONL event stream and
  renders a 6-panel dashboard: state machine / mic RMS sparkline+wake score /
  ASR transcript / streaming subconscious tokens / Hermes heartbeat / TTS
  progress bar. Watch your assistant think.

- **A knowledge-base hook.** If you maintain a personal Markdown wiki
  (Karpathy-style "LLM Wiki" — a folder of `.md` files under
  `vault/ref/` + `vault/wiki/`), point JARVIS at it and the agent can
  answer questions grounded in your own notes, not by making things up.
  The tool layer exposes `rag.search` (keyword match over the wiki) and
  `fs.read` (open the actual file) — what the agent does with them is up to
  whatever content you chose to put there.

## Architecture

```
    ┌───────────────────────────────────────────────────────────────┐
    │  Microphone  →  parec (PulseAudio) → 16 kHz PCM queue          │
    │                                  │                             │
    │          ┌─ openWakeWord (hey_jarvis_v0.1) + custom verifier ──┤
    │          └─ Silero VAD (end-of-speech)                         │
    │                                  │                             │
    │  IDLE → LISTENING → PROCESSING → RESPONDING → FOLLOW_UP_LISTEN │
    │                                  │                             │
    │                            Qwen3-ASR :8002                    │
    │                                  ↓                             │
    │                         Subconscious ──(tool_call)──┐          │
    │                  (SGLang Qwen3.6, :8000)            ↓          │
    │                                  │         Hermes Agent CLI    │
    │                                  │          ↓ via MCP :8005    │
    │                                  │  fs / rag / web / cf / ...  │
    │                                  ↓                             │
    │                         VoxCPM2 TTS :8003                     │
    │                                  │                             │
    │                               Speaker                          │
    └───────────────────────────────────────────────────────────────┘
```

5 systemd services run on a single Linux box:

| Service | Port | What it does |
|---|---|---|
| `sglang` | :8000 | Serves Qwen3.6-35B-A3B-FP8 (128K ctx, qwen3_coder tool parser) |
| `qwen3-asr-shim` | :8002 | FastAPI wrapper over `qwen_asr` Python API |
| `voxcpm2-tts` | :8003 | FastAPI streaming 48 kHz int16 PCM, zero-shot voice cloning |
| `hermes-shim` | :8004 | OpenAI `/v1/chat/completions` wrapper over Hermes Agent CLI |
| `mcp-server` | :8005 | MCP tool server (fs / web / rag / cf / system namespaces) |
| `jarvis-v3` | — | Main daemon: mic, wake, VAD, state machine, audio player, TUI event emitter |

## The terminal dashboard

`jarvis-tui` (which is `python tui/dashboard.py`) tails a JSONL event stream
emitted by the daemon and renders a live, six-panel [rich.Live](https://rich.readthedocs.io/)
TUI:

1. **State** — current state machine node + recent transitions, session id.
2. **Mic / Wake** — RMS in dBFS + 30-second sparkline + peak wake score.
3. **ASR** — last transcript with language detection + elapsed time.
4. **Subconscious** — the user message + streaming reply tokens + any tool
   call it issued.
5. **Hermes** — "running …Xs" with a pulse animation while the agent is in
   flight; shows final reply + success/fail status when done.
6. **TTS** — playback progress bar, characters remaining, elapsed.

Any subsystem error surfaces in the bottom panel with a timestamp and
component tag.

## Install (Linux server + USB mic/speaker)

**Hardware I've tested on:**
- Desktop: NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM), Ubuntu 24.04, CUDA 13.1
- USB audio: Plantronics Poly Sync 10 (other USB conference speakerphones
  should work — you need a device PulseAudio exposes with a 16 kHz
  `.mono-fallback` source to avoid AEC downsampling artefacts)
- Any quantisation of Qwen3.6-35B-A3B that fits in ≥ 50 GB VRAM
  (FP8 on a single Blackwell card; you could substitute Qwen3.6-14B or
  Llama-3.3-70B with minor config changes)

**Walk-through:**

```bash
# 1. Clone & venv
git clone https://github.com/CarverXx/jarvis-v3 ~/jarvis-v3
python3.12 -m venv ~/jarvis-v3-venv
source ~/jarvis-v3-venv/bin/activate

# 2. Install Python deps
pip install -r ~/jarvis-v3/requirements.txt

# 3. Models
# Qwen3.6-35B-A3B-FP8:  huggingface-cli download Qwen/Qwen3.6-35B-A3B-FP8
# Qwen3-ASR-1.7B:       huggingface-cli download Qwen/Qwen3-ASR-1.7B
# VoxCPM2:              huggingface-cli download openbmb/VoxCPM2

# 4. External agent
# Install Hermes Agent CLI: https://hermes.nousresearch.com
# Point ~/.hermes/config.yaml at your SGLang endpoint (http://127.0.0.1:8000/v1)

# 5. API keys (optional — only if you want web search / Cloudflare ops)
#    Replace REPLACE_ME with your own real key from tavily.com / dash.cloudflare.com
mkdir -p ~/.config/jarvis && chmod 700 ~/.config/jarvis
echo "REPLACE_ME_TAVILY_KEY"     > ~/.config/jarvis/tavily.key && chmod 600 $_
echo "REPLACE_ME_CLOUDFLARE_TOKEN" > ~/.config/jarvis/cf.token   && chmod 600 $_

# 6. Install systemd units
for f in ~/jarvis-v3/systemd/*.sample; do
    # Customize YOUR_USERNAME / YOUR_MIC_DEVICE / YOUR_SPEAKER_DEVICE
    # / REPLACE_ME_WITH_YOUR_SGLANG_API_KEY in each file first!
    base="$(basename "$f" .sample)"
    sudo install -m 644 "$f" "/etc/systemd/system/$base"
done
sudo systemctl daemon-reload

# 7. Start the stack
sudo systemctl enable --now sglang mcp-server qwen3-asr-shim voxcpm2-tts hermes-shim jarvis-v3

# 8. Run the TUI dashboard (in another SSH session)
sudo install -m 755 scripts/jarvis-tui /usr/local/bin/jarvis-tui   # once
jarvis-tui
```

Now say **"Hey Jarvis"** at the mic.

## Personalise (optional but recommended)

### 1. Voice cloning

Record 4–8 seconds of the voice you want JARVIS to speak in (could be yours,
or a character from a film / game if licensing permits). Save as
`.m4a`/`.wav`.

```bash
python scripts/trim_voice_ref.py --input ~/Downloads/my_voice.m4a
# writes a normalised, loudnorm'd assets/jarvis_voice_ref.wav
```

### 2. Wake-word custom verifier

Dramatically reduces false positives from non-you voices and boosts your own
wake recall.

```bash
# Record ~35 s of you saying "Hey Jarvis" 15-20 times (use Apple Voice Memos
# / Android Recorder / anything with clean background noise)
python scripts/split_wake_recording.py \
    --input ~/Downloads/hey_jarvis_batch.m4a \
    --out-dir assets/wake_samples/positive --n 15

# Record ~35 s of you saying 10 random things that AREN'T "Hey Jarvis"
python scripts/split_wake_recording.py \
    --input ~/Downloads/random_speech.m4a \
    --out-dir assets/wake_samples/negative --n 10

# Train the verifier
python scripts/train_wake_verifier.py
# writes assets/hey_jarvis_verifier.pkl — picked up automatically by daemon/wake_word.py
sudo systemctl restart jarvis-v3
```

### 3. Persona prompt

Two environment variables let you inject your own system prompts without
editing code:

- `JARVIS_SUBCONSCIOUS_PROMPT` — the subconscious persona (what JARVIS sounds
  like in conversation).
- `JARVIS_HERMES_VOICE_PROMPT` — the spoken-output style guide that every
  Hermes tool call is prepended with.

See [`docs/customize.md`](docs/customize.md) (TODO) for examples.

## Hardware / software requirements

| | Minimum | Tested |
|---|---|---|
| GPU VRAM | ~50 GB (SGLang Qwen3.6 + VoxCPM2 + Qwen3-ASR, all loaded simultaneously) | 96 GB (RTX PRO 6000 Blackwell) |
| CPU | 8 cores (wake+VAD is pure CPU) | 16 cores |
| RAM | 32 GB | 96 GB |
| OS | Linux with PipeWire/PulseAudio | Ubuntu 24.04.4 LTS |
| Python | 3.12 | 3.12.x |
| CUDA | 12.4+ | 13.1 |

A ≈ 30 B model is realistically the smallest thing that can reliably do
tool-calling well in Chinese. If you have less VRAM, swap the Qwen3.6 stack
for Qwen3-14B / Gemma-3-12B / Mistral-Small and accept the quality hit.

## Status

- ✅ Wake → ASR → Subconscious → optional Hermes → TTS end-to-end
- ✅ 5 systemd services auto-start and survive restart (tested SIGKILL + autoresume)
- ✅ openWakeWord custom verifier loaded, per-voice tuning
- ✅ VoxCPM2 zero-shot voice cloning, RMS stddev < 1 dB across turns
- ✅ TUI dashboard renders live in any terminal, handles malformed events
- ✅ Markdown wiki support via `rag.search` + `fs.read` (bring your own content)
- ✅ Tool execution for files / web search / Cloudflare / system state
- ✅ Comprehensive stress test: 97/97 passing (edge cases / concurrency / service crash recovery / 20-turn sessions)

**Not done yet (or not on by default):**
- Smart-home integration (no HomeAssistant bridge wired up)
- Calendar / email / iMessage tools
- Mobile companion — runs on a Linux host + USB mic, no phone app
- Multi-speaker distinction (wake fires on anyone saying the wake word; the
  custom verifier helps but doesn't fully lock out other voices)

## Repository layout

```
jarvis-v3/
├── daemon/                  # Main Python process (runs on jarvis-v3.service)
│   ├── satellite.py         # State machine, mic reader, audio player
│   ├── wake_word.py         # openWakeWord + custom verifier loader
│   ├── vad.py               # Silero VAD wrapper
│   ├── subconscious.py      # SGLang-based conversational LLM + tool routing
│   ├── tools.py             # invoke_hermes tool schema + execution
│   ├── backend_client.py    # HTTP clients for the 3 shims
│   ├── events.py            # JSONL event emitter for the TUI
│   └── state.py             # State enum
├── services/                # FastAPI shims, each its own systemd unit
│   ├── hermes_shim.py       # :8004 — Hermes Agent CLI wrapper
│   ├── qwen3_asr_shim.py    # :8002 — Qwen3-ASR wrapper
│   └── voxcpm2_tts.py       # :8003 — VoxCPM2 streaming TTS
├── tui/dashboard.py         # rich.Live terminal dashboard
├── scripts/
│   ├── trim_voice_ref.py    # Extract a clean voice reference from any recording
│   ├── split_wake_recording.py  # Slice a batch recording into wake-word training samples
│   ├── regen_assets.py      # Regenerate the wake chime and waiting beep
│   └── sglang-launcher.sh   # systemd launcher for SGLang with JSON config
├── systemd/                 # *.service.sample templates — customise before installing
├── assets/                  # (gitignored) audio assets, voice refs, verifier pkl
├── docs/                    # Landing page, images, customization guides
├── config.py                # Central config (env-driven)
└── requirements.txt
```

## Acknowledgements

- [openWakeWord](https://github.com/dscripka/openWakeWord) — wake detection + custom verifier
- [Silero VAD](https://github.com/snakers4/silero-vad) — end-of-speech detection
- [Qwen team](https://github.com/QwenLM) — Qwen3-ASR and Qwen3.6 LLM
- [OpenBMB VoxCPM](https://github.com/OpenBMB/VoxCPM) — streaming voice-cloning TTS
- [SGLang](https://github.com/sgl-project/sglang) — high-throughput LLM serving with tool parsing
- [Nous Research · Hermes Agent CLI](https://hermes.nousresearch.com) — the agentic backend
- [Wyoming Satellite](https://github.com/rhasspy/wyoming-satellite) — half-duplex state machine reference
- [Home Assistant · AssistSatelliteState](https://github.com/home-assistant/core) — state-naming convention

## License

MIT — see [LICENSE](LICENSE).

This project is a personal build for my home setup. Open-sourced in the hope
that the dual-brain pattern and the half-duplex audio plumbing are useful to
anyone else building local voice assistants. Issues / PRs welcome but I
maintain this part-time.
