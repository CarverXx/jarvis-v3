"""
hermes_shim — OpenAI-compatible /v1/chat/completions over Hermes Agent CLI.

Listens on :8004. Daemon (jarvis-v3) calls this with `X-Session-Id: <uuid>`
and OpenAI-style `messages[]`. We extract the last user message, shell out to
`hermes chat -q <msg> -Q`, and use Hermes' native `--resume <hermes_session>`
to preserve multi-turn state inside Hermes itself (tool calls, memory, skills).

We map our caller's X-Session-Id to a Hermes session id in memory, so each
voice-assistant conversation stays consistent across turns.

Adapted from jarvis-v2 server/jarvis/mcp/tools/hermes.py (_parse_hermes_stdout,
subprocess pattern).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

logger = logging.getLogger("hermes-shim")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

HERMES_BIN = os.path.expanduser(
    os.environ.get("HERMES_BIN", "~/.local/bin/hermes")
)
DEFAULT_TIMEOUT_S = int(os.environ.get("HERMES_TIMEOUT_S", "60"))
# Keep Hermes replies SHORT — they go through a second Qwen3.6 "subconscious"
# pass to turn them into natural-sounding speech. A 4000-char markdown dump
# confuses the summariser. 600 chars ≈ 200 Chinese tokens ≈ plenty.
MAX_RESPONSE_CHARS = int(os.environ.get("HERMES_MAX_REPLY_CHARS", "600"))

# Voice-assistant system prompt injected onto EVERY first-turn call — forces
# Hermes to answer in plain prose (no markdown/tables/emoji) and to stay short.
# Without this Hermes's agent loop loves to emit formatted markdown + thinking
# narration, which then confuses the Qwen3.6 summarisation layer.
VOICE_SYSTEM_PROMPT = os.environ.get("JARVIS_HERMES_VOICE_PROMPT") or """你是家庭语音助手的执行层。**内部可以多步推理 + 多次调工具**, 但最终输出给用户的那段话是纯文本 2-3 句。

【关键工作流程】(内部你必须做到)
1. 看到问题 → 挑合适的工具 → 调一次。
2. 如果第一次工具返回**只是索引/摘要/片段** (例如 rag.search 返回 Home.md 的目录项), 那就**继续深入**:
   - rag.search 返回 path 后 → **再调 fs.read 读那个 path 的实际内容**
   - web.search 返回 URL 后 → 直接看摘要 (不要念 URL 给用户)
   - 结果不足就换关键词再 rag.search 一次。**重要: 允许 3-5 次工具调用, 不要过早下"查不到"的结论。**
3. 得到足够信息后, **改写**成 2-3 句中文口语化句子。

【最终输出格式】(仅约束最终输出, 不约束思考过程)
- 简体中文纯文本, 2-3 句。
- 不要 markdown / 表格 / 项目符号 / emoji / 代码块 / URL / 原始 JSON / 路径。
- 数字用阿拉伯数字, 日期用 "4 月 19 日" 格式。

【工具与场景】
- 天气 / 新闻 / 股价 / 汇率 → web.search
- 个人知识库 (家庭/设备/项目/财务/健康/证件) → rag.search 搜, 然后 **fs.read** 读找到的 .md 文件
- 记笔记 → raphael.save_note / raphael.append_log
- Cloudflare → cf.list_zones / cf.list_tunnels / cf.list_dns_records
- **时间/日期** → **必须** 调 system.date (不要凭训练数据猜, 日期会错)
- 服务状态 / 音量 → system.list_services / system.volume
- 读文件 / 列目录 → fs.read / fs.list

❌ 没有的工具 (直接 1 句说不行): 日历, 邮件, iMessage/微信, 智能家居, 音乐播放, 电话

【反模式】
❌ rag.search 返回了 CLAUDE.md 目录 → 直接说"没记录"  (错! 要 fs.read 打开那个指向的文件)
❌ 凭记忆回答时间 (错! 必须调 system.date)
❌ 输出"让我查一下"/"现在我要" (错! 思考不要说出来, 直接给答案)
❌ 输出"用户要求..."/"我应该..."/"我需要..." (错! 直接给答案, 不要复述用户问题)
❌ 文件里有 `[REDACTED]` → 说"无法读取" (错! [REDACTED] 只替换了密码/卡号几个字段, 其他内容如公司名/地址/进度全部正常可读, 继续 fs.read 拿信息)
❌ 文件开头有 "脱敏版本" banner 或 "原始数据在 ..." 字样 → 说"文件不在本地"  (错! banner 只是个说明, 正文照样读, 正文本身是完整的, 只是 SECRET 字段被 [REDACTED] 替换)
❌ 复述"RAG 的全量索引还没建好" / "Auto-repaired tool name" / "信息已经足够全面了" (错! 这些是内部日志, 不要说给用户)

示例:
用户"今天几号" → system.date → "今天是 4 月 19 日, 星期日"
用户"家人几个" → rag.search("家庭成员") 得到片段 → fs.read(知识库里的文件) 拿全文 → "记录了 X 位家人"
用户"上海天气" → web.search → "上海今天多云 20 度"
用户"打开日历" → "我暂时没接日历, 没法查"
"""

# our_session_id (uuid sent by daemon) → hermes session id
# Kept in memory only; voice-assistant session lifetime = daemon uptime.
_session_map: dict[str, str] = {}

# Match a `session_id: xxx` banner line from hermes stdout.
_SESSION_BANNER_RE = re.compile(r"^session_id:\s*(\S+)", re.IGNORECASE)


# ---------- OpenAI-compatible schemas (subset we actually honor) ----------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None           # ignored — Hermes has its own backend
    messages: list[ChatMessage]
    stream: bool = False               # ignored — Hermes is non-streaming
    temperature: float | None = None   # ignored
    max_tokens: int | None = None      # ignored


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = "hermes-agent"
    choices: list[ChatChoice]
    usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})


# ---------- Hermes CLI helpers ----------
def _parse_hermes_output(stdout: str, stderr: str) -> tuple[str | None, str]:
    """Returns (hermes_session_id_if_any, cleaned_reply_text).

    Hermes CLI with `--pass-session-id` prints `session_id: <uuid>` on STDERR
    (observed 2026-04-19), while the assistant reply goes to STDOUT. We scan
    stderr for the banner and keep stdout cleaned of warning lines."""
    hermes_sid: str | None = None
    for line in stderr.splitlines():
        m = _SESSION_BANNER_RE.match(line.strip())
        if m:
            hermes_sid = m.group(1)
            break

    kept: list[str] = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        if (s.startswith("⚠") or s.startswith("↻")
                or "auxiliary LLM" in s
                or "Run `hermes" in s
                or "Resumed session" in s):
            continue
        kept.append(line)
    reply = "\n".join(kept).strip()
    reply = _clean_for_tts(reply)
    return hermes_sid, reply


# Strip markdown, code blocks, reasoning narrations — anything that makes TTS
# awkward. Hermes agent's default output is heavy on tables / bullets / emoji
# and occasionally includes thinking aloud like "Now let me compile the
# report". We neutralise it here so the subconscious summariser gets clean
# prose to work with.
_MD_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$|^\s*\|?[-:\s|]+\|?\s*$")
_MD_BULLET = re.compile(r"^\s*[-*•+]\s+|^\s*\d+\.\s+")
_MD_HEADING = re.compile(r"^\s*#{1,6}\s+")
_CODE_FENCE = re.compile(r"```[^\n]*\n([\s\S]*?)```", re.MULTILINE)
_INLINE_CODE = re.compile(r"`([^`]+)`")
_URL_PATTERN = re.compile(r"https?://\S+")
_EMOJI_RANGE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoticons, transport
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)
# Planning/thinking SENTENCE — a whole sentence (up to 。/!/?) that starts
# with one of these patterns. We kill the whole meta-sentence but preserve
# anything that comes AFTER it on the same line (the real answer).
# Trigger vocabulary — if any of these appear INSIDE a sentence, the whole
# sentence is considered "thinking aloud" and dropped. Much more robust than
# trying to match the leading position.
_TRIGGER_WORDS = re.compile(
    r"(?:Now let me|Let me(?: |,)|First,|Here is|Here's|Okay,|OK,|"
    r"综合以上|综合来看|总结一下|让我(?:查|看|来|搜|分析|整理|确认)|我来(?:查|看|分析|汇总)|"
    r"现在我(?:需要|要|来|应该|可以)|首先[,，]|接下来[,，]|"
    r"用户(?:要求|希望|问|想|需要)|我应该(?:使用|调用|尝试)|我需要(?:使用|调用|查)|"
    r"我已经(?:拿到|找到|收集|搜索|分析|匹配|整合|确认|查到)|"
    r"根据(?:搜索|结果|信息|片段|索引)|综合.*?信息|"
    r"信息(?:已经?)?(?:足够|足以|充足|完整)|可以回答(?:了|你|用户)|"
    r"已经(?:拿到|找到|收集|搜索|分析|匹配|整合|确认|查到).{0,5}(?:信息|数据|结果)|"
    r"拿到.{0,8}数据|得到.{0,8}信息|收集到.{0,8}信息|"
    r"现在(?:整合|让我|可以整理|汇总|总结)|整合成|汇总成|"
    r"RAG.{0,10}索引|rag.{0,10}search|全量索引|索引.{0,5}(?:还没|暂未|未)?.{0,5}(?:建|构)|"
    r"只能靠.{0,5}(?:匹配|片段)|从.{0,5}片段.{0,5}(?:可|能)?看|"
    r"再确认一下|让我?再确认|最终.{0,5}结论)",
    re.IGNORECASE,
)
# Exclamation stand-alone leaders — kill ONLY the exclamation, not what follows.
_META_EXCLAIM = re.compile(
    r"(?:^|(?<=[\n。]))\s*(?:完美|太好了|好了|棒极了|Perfect)[!！][\s]*",
    re.IGNORECASE,
)


def _strip_meta_sentences(text: str) -> str:
    """Split text on [。！？] and drop any sentence whose trigger is near
    the start (first ~20 chars). This preserves sentences where the
    trigger word happens to appear mid-sentence in a legitimate answer."""
    parts = re.split(r"(?<=[。！？\n])", text)
    kept: list[str] = []
    for p in parts:
        m = _TRIGGER_WORDS.search(p)
        if m:
            # Compute offset in trimmed sentence (ignore leading whitespace).
            offset = m.start() - (len(p) - len(p.lstrip()))
            if offset < 25:  # trigger appears near sentence start → meta
                continue
        kept.append(p)
    return "".join(kept)
# Bare Auto-repaired lines — kill the whole line (no content to preserve).
_AUTO_REPAIR_NOISE = re.compile(
    r"Auto-repaired tool name:\s*\S+?\s*->\s*\S+",
    re.IGNORECASE,
)
# Stray internal tokens (tokenizer artifacts like "[[audio_as_voice]]").
_JUNK_TOKENS = re.compile(r"\[\[[a-z_]+\]\]", re.IGNORECASE)
# Lines that are 100% meta/planning — drop them entirely.
_META_LINES = re.compile(
    r"^(?:No response requested|Let me .{0,60}$|I should .{0,60}$|"
    r"I'll .{0,60}$|我应该.{0,40}$|我需要.{0,40}$|我会.{0,40}$|我来.{0,40}$|"
    r"Auto-repaired tool name.*$|.*tool_name.*->.*$)\s*\.?\s*$",
    re.IGNORECASE,
)


def _clean_for_tts(text: str) -> str:
    if not text:
        return text
    # Strip code fences (preserve inner content as plain text prefix "代码略")
    text = _CODE_FENCE.sub("[代码略]", text)
    text = _INLINE_CODE.sub(r"\1", text)
    # Strip bold/italic markdown (** and * around text) — keeps inner text,
    # removes stars that TTS would read as "星号".
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"(?<![A-Za-z])\*([^*\n]+)\*(?![A-Za-z])", r"\1", text)  # *italic* but not C*  *)
    # Kill specific noise FIRST (document-wide, not per-line).
    text = _AUTO_REPAIR_NOISE.sub("", text)
    text = _JUNK_TOKENS.sub("", text)
    # Drop meta exclamations like "完美!" at sentence start.
    text = _META_EXCLAIM.sub("", text)
    # Then drop any whole sentence that contains a thinking/meta trigger.
    text = _strip_meta_sentences(text)
    # Remove URL noise
    text = _URL_PATTERN.sub("", text)
    # Remove emoji
    text = _EMOJI_RANGE.sub("", text)
    # Line-by-line cleanup: drop table rows, bullet markers, heading hashes
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _MD_TABLE_LINE.match(stripped):
            continue
        # Drop pure meta/planning lines entirely (e.g. "No response requested.")
        if _META_LINES.match(stripped):
            continue
        stripped = _MD_BULLET.sub("", stripped)
        stripped = _MD_HEADING.sub("", stripped)
        if stripped:
            out_lines.append(stripped)
    # Collapse to single paragraph — voice output doesn't need line breaks.
    cleaned = " ".join(out_lines)
    # Drop horizontal-rule artifacts + collapse whitespace.
    cleaned = re.sub(r"---+", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


async def _call_hermes(
    prompt: str,
    hermes_session: str | None,
    timeout_s: int,
) -> tuple[str | None, str, float, int]:
    """Shell out to hermes CLI. Returns (hermes_session_id, reply, elapsed_s, exit_code)."""
    if not os.path.exists(HERMES_BIN):
        raise HTTPException(status_code=500, detail=f"hermes binary not found at {HERMES_BIN}")

    env = dict(os.environ)
    env.setdefault("PATH", f"{os.path.expanduser('~/.local/bin')}:{env.get('PATH', '')}")

    # --pass-session-id and --resume are GLOBAL flags (before subcommand).
    # --pass-session-id makes hermes print `session_id: <uuid>` on stdout,
    # which _parse_hermes_output extracts so we can --resume on the next turn.
    args = [HERMES_BIN, "--pass-session-id"]
    if hermes_session:
        args += ["--resume", hermes_session]
    args += ["chat", "-q", prompt, "-Q", "--source", "jarvis-v3-shim"]

    t0 = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except Exception as e:
        logger.exception("subprocess spawn failed")
        raise HTTPException(status_code=500, detail=f"subprocess spawn failed: {e}")

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(status_code=504, detail=f"hermes timeout after {timeout_s}s")

    elapsed = time.monotonic() - t0
    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")

    if proc.returncode != 0:
        logger.warning("hermes exit=%d stderr=%s", proc.returncode, stderr[:400])
        raise HTTPException(
            status_code=502,
            detail=f"hermes exit {proc.returncode}: {stderr[:200].strip()}",
        )

    sid, reply = _parse_hermes_output(stdout, stderr)
    if not reply:
        reply = "(hermes 无回复)"
    if len(reply) > MAX_RESPONSE_CHARS:
        reply = reply[: MAX_RESPONSE_CHARS - 20].rstrip() + "……(已截断)"

    return sid, reply, elapsed, proc.returncode


# ---------- FastAPI app ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(HERMES_BIN):
        logger.warning("hermes binary %s not found — /v1/chat/completions will 500", HERMES_BIN)
    else:
        logger.info("hermes binary: %s", HERMES_BIN)
    logger.info("timeout=%ds max_reply_chars=%d", DEFAULT_TIMEOUT_S, MAX_RESPONSE_CHARS)
    yield
    logger.info("shutting down; dropped %d session mappings", len(_session_map))


app = FastAPI(title="hermes-shim", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "hermes_bin_exists": os.path.exists(HERMES_BIN),
        "active_sessions": len(_session_map),
    }


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    req: ChatRequest,
    x_session_id: str | None = Header(None, alias="X-Session-Id"),
):
    # Only the last user message goes into Hermes — the rest is already in
    # Hermes' resumed session state. If caller provides no X-Session-Id, we
    # treat each call as a new session (stateless fallback).
    last_user = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        None,
    )
    if not last_user or not last_user.strip():
        raise HTTPException(status_code=400, detail="no user message in request")

    hermes_session = _session_map.get(x_session_id) if x_session_id else None

    # Hermes CLI has no --system flag; inject system prompt as a prefix on
    # the FIRST turn only. Subsequent turns rely on --resume carrying the
    # system context via Hermes' internal session state.
    if hermes_session is None:
        caller_system = next(
            (m.content for m in req.messages if m.role == "system"), None
        )
        # Always prepend VOICE_SYSTEM_PROMPT (markdown/tts discipline) —
        # then append any caller-provided system prompt. This ensures even
        # fresh sessions from subconscious tool-calls inherit voice rules.
        combined_system = VOICE_SYSTEM_PROMPT
        if caller_system and caller_system.strip():
            combined_system = combined_system + "\n\n" + caller_system.strip()
        last_user = (
            f"[系统指令(必须严格遵守,不要复述)]:\n{combined_system}\n\n"
            f"[用户说]: {last_user}"
        )

    logger.info(
        "chat x_sid=%s hermes_sid=%s msg=%r",
        x_session_id[:8] + "…" if x_session_id else None,
        hermes_session[:8] + "…" if hermes_session else None,
        last_user[:80],
    )

    new_sid, reply, elapsed, _ = await _call_hermes(
        last_user, hermes_session, DEFAULT_TIMEOUT_S
    )

    # Capture hermes session_id on first turn so --resume works next time.
    if x_session_id and new_sid and not hermes_session:
        _session_map[x_session_id] = new_sid
        logger.info("bound x_sid=%s… → hermes_sid=%s…", x_session_id[:8], new_sid[:8])

    logger.info("reply (%.1fs): %r", elapsed, reply[:120])

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=reply))],
    )


@app.post("/v1/sessions/{x_session_id}/reset")
async def reset_session(x_session_id: str):
    """Manual session reset — drops our mapping so next call starts a fresh
    Hermes session. Hermes-side session keeps existing (not deleted) but is
    no longer addressable from this shim."""
    existed = _session_map.pop(x_session_id, None)
    return {"ok": True, "reset": existed is not None}


if __name__ == "__main__":
    port = int(os.environ.get("HERMES_SHIM_PORT", "8004"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
