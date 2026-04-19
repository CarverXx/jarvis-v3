"""
JARVIS v3 Terminal TUI — live dashboard for the daemon.

Tails `~/jarvis-v3-logs/events.jsonl` (see `daemon/events.py`) and renders:

    ┌─ State ───────────┬─ Mic / Wake ──────┐
    │ …                 │ …                 │
    ├───────────────────┴───────────────────┤
    │ ASR (你说的)                           │
    ├───────────────────────────────────────┤
    │ 潜意识 stream (SGLang Qwen3.6)         │
    ├───────────────────────────────────────┤
    │ 主意识 Hermes                          │
    ├───────────────────────────────────────┤
    │ TTS                                    │
    ├───────────────────────────────────────┤
    │ 错误/警告 tail                          │
    └───────────────────────────────────────┘

Launch (on totodile):
    ~/jarvis-v3-venv/bin/python ~/jarvis-v3-repo/tui/dashboard.py

Ctrl+C quits. Terminal resize auto-reflows. rich.Live throttled to 10 FPS.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def esc(s: str) -> str:
    """Escape brackets in user-derived content so Rich doesn't parse them
    as markup tags. All ASR / Hermes / TTS / tool content comes from
    untrusted sources and may contain `[...]` that Rich would try to
    interpret (e.g. `[red]`, `[/bold]`, `[test]`)."""
    return escape(s or "")

EVENTS_PATH = os.environ.get(
    "JARVIS_EVENTS_PATH",
    os.path.expanduser("~/jarvis-v3-logs/events.jsonl"),
)

SPARK_CHARS = "▁▂▃▄▅▆▇█"
MIC_HISTORY_SAMPLES = 80           # sparkline columns
WAKE_HISTORY_SECONDS = 30
STATE_HISTORY = 8
ERROR_TAIL = 6


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _spark(values: list[float], lo: float = -70.0, hi: float = -10.0) -> str:
    """Turn a list of dBFS floats into a unicode sparkline string."""
    if not values:
        return "—"
    span = hi - lo
    out = []
    for v in values:
        t = max(0.0, min(1.0, (v - lo) / span))
        out.append(SPARK_CHARS[int(t * (len(SPARK_CHARS) - 1))])
    return "".join(out)


class DashboardState:
    """Accumulates events into panel-ready derived state.
    No rendering logic lives here — render() just reads it."""

    def __init__(self):
        self.session_id: str = "—"
        self.current_state: str = "IDLE"
        self.state_history: deque[tuple[float, str]] = deque(maxlen=STATE_HISTORY)

        self.mic_rms_dbfs: float = -90.0
        self.mic_peak: float = 0.0
        self.mic_history: deque[float] = deque(maxlen=MIC_HISTORY_SAMPLES)

        self.wake_score_peak_recent: float = 0.0
        self.wake_peak_at: float = 0.0

        # ASR
        self.asr_text: str = "—"
        self.asr_lang: str = ""
        self.asr_elapsed_s: float = 0.0
        self.asr_last_ts: float = 0.0
        self.asr_in_flight_since: float = 0.0

        # Subconscious stream — cleared on each sub_start
        self.sub_user_text: str = "—"
        self.sub_buf: list[str] = []
        self.sub_tool: dict | None = None
        self.sub_last_done: str = ""
        self.sub_last_done_ts: float = 0.0
        self.sub_in_flight_since: float = 0.0

        # Hermes
        self.hermes_task: str = ""
        self.hermes_start_ts: float = 0.0
        self.hermes_last_tick: float = 0.0
        self.hermes_last_reply: str = ""
        self.hermes_last_ok: bool | None = None
        self.hermes_last_elapsed: float = 0.0
        self.hermes_running: bool = False

        # TTS
        self.tts_text: str = ""
        self.tts_chars: int = 0
        self.tts_start_ts: float = 0.0
        self.tts_bytes: int = 0
        self.tts_elapsed: float = 0.0
        self.tts_running: bool = False
        self.tts_last_elapsed: float = 0.0

        # Errors
        self.errors: deque[tuple[float, str, str]] = deque(maxlen=ERROR_TAIL)

    # ---------- event application ----------
    def apply(self, ev: dict):
        t = float(ev.get("t", time.time()))
        et = ev.get("type", "")
        if et == "session":
            self.session_id = ev.get("session_id", self.session_id)
        elif et == "state":
            to = ev.get("to", "?")
            if to != self.current_state:
                self.state_history.append((t, self.current_state))
            self.current_state = to
        elif et == "mic":
            # Defensive: unknown / None / non-numeric values shouldn't crash.
            rms = ev.get("rms_dbfs")
            peak = ev.get("peak")
            try:
                self.mic_rms_dbfs = float(rms) if rms is not None else -90.0
            except (TypeError, ValueError):
                self.mic_rms_dbfs = -90.0
            try:
                self.mic_peak = float(peak) if peak is not None else 0.0
            except (TypeError, ValueError):
                self.mic_peak = 0.0
            # NaN sanity — a NaN here would corrupt the sparkline.
            import math
            if math.isnan(self.mic_rms_dbfs):
                self.mic_rms_dbfs = -90.0
            self.mic_history.append(self.mic_rms_dbfs)
        elif et == "wake":
            score = float(ev.get("score", 0.0))
            if t - self.wake_peak_at > WAKE_HISTORY_SECONDS:
                self.wake_score_peak_recent = 0.0
            if score > self.wake_score_peak_recent:
                self.wake_score_peak_recent = score
                self.wake_peak_at = t
        elif et == "asr_start":
            self.asr_in_flight_since = t
        elif et == "asr_done":
            self.asr_text = ev.get("text", "")
            self.asr_lang = ev.get("lang", "") or ""
            self.asr_elapsed_s = float(ev.get("elapsed_s", 0.0))
            self.asr_last_ts = t
            self.asr_in_flight_since = 0.0
        elif et == "sub_start":
            self.sub_user_text = ev.get("user_text", "")
            self.sub_buf.clear()
            self.sub_tool = None
            self.sub_in_flight_since = t
        elif et == "sub_token":
            piece = ev.get("delta", "")
            if piece:
                self.sub_buf.append(piece)
        elif et == "sub_tool":
            self.sub_tool = {
                "tool": ev.get("tool", "?"),
                "args": ev.get("args", {}),
            }
        elif et == "sub_done":
            self.sub_last_done = ev.get("content", "")
            self.sub_last_done_ts = t
            self.sub_in_flight_since = 0.0
        elif et == "hermes_start":
            self.hermes_running = True
            self.hermes_task = ev.get("task", "")
            self.hermes_start_ts = t
            self.hermes_last_tick = t
        elif et == "hermes_tick":
            self.hermes_last_tick = t
        elif et == "hermes_done":
            self.hermes_running = False
            self.hermes_last_reply = ev.get("reply", "")
            self.hermes_last_ok = bool(ev.get("ok", True))
            self.hermes_last_elapsed = float(ev.get("elapsed_s", 0.0))
        elif et == "tts_start":
            self.tts_running = True
            self.tts_text = ev.get("text", "")
            self.tts_chars = int(ev.get("chars", 0))
            self.tts_start_ts = t
            self.tts_bytes = 0
        elif et == "tts_playing":
            self.tts_bytes = int(ev.get("bytes_played", 0))
            self.tts_elapsed = float(ev.get("elapsed_s", 0.0))
        elif et == "tts_done":
            self.tts_running = False
            self.tts_last_elapsed = float(ev.get("elapsed_s", 0.0))
        elif et == "error":
            self.errors.append((t, ev.get("where", "?"), ev.get("msg", "")))


# --------------- render ---------------
STATE_STYLES = {
    "IDLE":              "dim",
    "LISTENING":         "bold cyan",
    "PROCESSING":        "bold yellow",
    "RESPONDING":        "bold green",
    "FOLLOW_UP_LISTEN":  "bold blue",
    "ERROR":             "bold red",
}


def _panel_state(s: DashboardState) -> Panel:
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(justify="right", style="dim")
    tbl.add_column()

    state_style = STATE_STYLES.get(s.current_state, "bold white")
    tbl.add_row("当前状态", Text(s.current_state, style=state_style))
    hist = " → ".join(h[1] for h in list(s.state_history)[-STATE_HISTORY:]) or "—"
    hist_display = f"{hist} → [bold]{s.current_state}[/bold]"
    tbl.add_row("历史", Text.from_markup(hist_display))
    tbl.add_row("session", Text(s.session_id, style="dim"))

    events_age = time.time() - max(
        s.asr_last_ts, s.sub_last_done_ts, s.tts_start_ts, 0.0
    )
    stale = events_age > 60 and s.current_state == "IDLE"
    tbl.add_row(
        "idle 时长",
        Text(f"{events_age:.0f}s", style="red" if stale else "dim"),
    )
    return Panel(tbl, title="[bold]State[/bold]", border_style="cyan")


def _panel_mic(s: DashboardState) -> Panel:
    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(justify="right", style="dim")
    tbl.add_column()

    rms_style = (
        "red" if s.mic_rms_dbfs > -20
        else "green" if s.mic_rms_dbfs > -50
        else "dim"
    )
    tbl.add_row("RMS", Text(f"{s.mic_rms_dbfs:+.1f} dBFS", style=rms_style))
    spark = _spark(list(s.mic_history))
    tbl.add_row("波形", Text(spark, style="cyan"))
    wake_age = time.time() - s.wake_peak_at if s.wake_peak_at else 999
    wake_style = (
        "bold yellow" if s.wake_score_peak_recent >= 0.5 and wake_age < 10
        else "dim"
    )
    tbl.add_row(
        "wake peak",
        Text(f"{s.wake_score_peak_recent:.3f} ({wake_age:.0f}s ago)", style=wake_style),
    )
    return Panel(tbl, title="[bold]Mic / Wake[/bold]", border_style="magenta")


def _panel_asr(s: DashboardState) -> Panel:
    if s.asr_in_flight_since:
        body = Text(f"...录音处理中 ({time.time() - s.asr_in_flight_since:.1f}s)", style="yellow")
    elif s.asr_last_ts:
        ts = _fmt_time(s.asr_last_ts)
        asr_text_e = esc(s.asr_text) if s.asr_text else "(空)"
        lang_e = f" [{esc(s.asr_lang)}]" if s.asr_lang else ""
        body = Text.from_markup(
            f"[dim]{ts}[/dim] [bold]{asr_text_e}[/bold][dim]{lang_e} ({s.asr_elapsed_s:.2f}s)[/dim]"
        )
    else:
        body = Text("—", style="dim")
    return Panel(body, title="[bold]ASR (你说的)[/bold]", border_style="blue")


def _panel_subconscious(s: DashboardState) -> Panel:
    lines: list[Text] = []
    if s.sub_user_text and s.sub_user_text != "—":
        lines.append(Text.from_markup(f"[dim]USER:[/dim] {esc(s.sub_user_text)}"))
    # Cap sub_buf rendering at ~600 chars to prevent TUI overflow on very
    # long streamed replies.
    buf = "".join(s.sub_buf).strip()
    if buf:
        buf_show = buf if len(buf) <= 600 else buf[:600] + "…"
        lines.append(Text.from_markup(f"[green]AGENT:[/green] {esc(buf_show)}"))
    if s.sub_tool:
        tool = s.sub_tool.get("tool", "?")
        args = s.sub_tool.get("args", {})
        task = args.get("task", "") if isinstance(args, dict) else ""
        lines.append(Text.from_markup(
            f"[bold yellow]tool_call:[/bold yellow] {esc(tool)}(task={esc(repr(task))})"
        ))
    if not lines:
        if s.sub_last_done:
            lines.append(Text.from_markup(f"[dim]上轮:[/dim] {esc(s.sub_last_done)}"))
        else:
            lines.append(Text("—", style="dim"))
    if s.sub_in_flight_since:
        spinner = "·" * (int(time.time() * 4) % 4 + 1)
        lines.append(Text(f"推理中 {spinner} {time.time() - s.sub_in_flight_since:.1f}s", style="yellow"))
    return Panel(Group(*lines), title="[bold]潜意识 stream (SGLang Qwen3.6)[/bold]", border_style="green")


def _panel_hermes(s: DashboardState) -> Panel:
    if s.hermes_running:
        elapsed = time.time() - s.hermes_start_ts
        dots = "●" * min(15, int(elapsed / 2))
        body = Text.from_markup(
            f"[bold yellow]● 运行中 {elapsed:.1f}s[/bold yellow]\n"
            f"task=[bold]{esc(s.hermes_task)}[/bold]\n"
            f"[yellow]{dots}[/yellow]"
        )
    elif s.hermes_last_ok is not None:
        status = "[bold green]✓[/bold green]" if s.hermes_last_ok else "[bold red]✗[/bold red]"
        reply = s.hermes_last_reply[:200] + ("…" if len(s.hermes_last_reply) > 200 else "")
        body = Text.from_markup(
            f"{status} 上次 {s.hermes_last_elapsed:.1f}s\n"
            f"[dim]task:[/dim] {esc(s.hermes_task[:100])}\n"
            f"[dim]reply:[/dim] {esc(reply)}"
        )
    else:
        body = Text("—", style="dim")
    return Panel(body, title="[bold]主意识 Hermes[/bold]", border_style="yellow")


def _panel_tts(s: DashboardState) -> Panel:
    lines: list[Text] = []
    if s.tts_running:
        pct = 0
        if s.tts_chars > 0:
            # rough heuristic: ~25 char/s at 48kHz voxcpm
            est_total_s = s.tts_chars / 4.5
            pct = int(min(100, 100 * s.tts_elapsed / max(0.1, est_total_s)))
        filled = int(pct / 5)
        bar = "▰" * filled + "░" * (20 - filled)
        lines.append(Text.from_markup(
            f"[bold green]播放中[/bold green] {s.tts_elapsed:.1f}s  "
            f"[green]{bar}[/green] {pct}%"
        ))
        lines.append(Text.from_markup(f"[dim]text:[/dim] {esc(s.tts_text[:120])}"))
    elif s.tts_last_elapsed > 0:
        lines.append(Text.from_markup(
            f"[dim]上次播放完成 {s.tts_last_elapsed:.2f}s[/dim]"
        ))
        if s.tts_text:
            lines.append(Text.from_markup(f"[dim]text:[/dim] {esc(s.tts_text[:120])}"))
    else:
        lines.append(Text("—", style="dim"))
    return Panel(Group(*lines), title="[bold]TTS (VoxCPM + JARVIS ref)[/bold]", border_style="bright_green")


def _panel_errors(s: DashboardState) -> Panel:
    if not s.errors:
        body = Text("—", style="dim")
    else:
        rows: list[Text] = []
        for t, where, msg in list(s.errors)[-ERROR_TAIL:]:
            rows.append(Text.from_markup(
                f"[dim]{_fmt_time(t)}[/dim] [red]{esc(where)}[/red] {esc(msg[:140])}"
            ))
        body = Group(*rows)
    return Panel(body, title="[bold]错误/警告[/bold]", border_style="red")


def _build_layout(s: DashboardState) -> Layout:
    root = Layout()
    root.split_column(
        Layout(name="top", size=8),
        Layout(name="asr", size=4),
        Layout(name="sub", size=7),
        Layout(name="hermes", size=6),
        Layout(name="tts", size=5),
        Layout(name="err", size=8),
    )
    top = root["top"]
    top.split_row(
        Layout(_panel_state(s), name="state"),
        Layout(_panel_mic(s), name="mic"),
    )
    root["asr"].update(_panel_asr(s))
    root["sub"].update(_panel_subconscious(s))
    root["hermes"].update(_panel_hermes(s))
    root["tts"].update(_panel_tts(s))
    root["err"].update(_panel_errors(s))
    return root


# --------------- tail ---------------
async def tail_events(path: str, state: DashboardState, catch_up: int = 200):
    """Open (or wait for) the events file; seek to last `catch_up` lines; tail."""
    # Wait until the file exists (daemon may not have started yet)
    while not os.path.exists(path):
        await asyncio.sleep(0.5)

    # Seek to last N lines for catch-up context.
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # Read up to the last ~64KB for catch-up.
            chunk_size = min(size, 65536)
            f.seek(size - chunk_size, os.SEEK_SET)
            tail = f.read().decode("utf-8", errors="replace").splitlines()
        for line in tail[-catch_up:]:
            if not line.strip():
                continue
            try:
                state.apply(json.loads(line))
            except Exception:
                continue
    except Exception:
        pass

    # Live tail: poll the file for new data. Keep the byte offset we read up to.
    last_size = os.path.getsize(path) if os.path.exists(path) else 0
    buf = ""
    while True:
        try:
            cur_size = os.path.getsize(path)
        except FileNotFoundError:
            await asyncio.sleep(0.5)
            continue
        if cur_size < last_size:
            # Log rotated — reset
            last_size = 0
            buf = ""
        if cur_size > last_size:
            with open(path, "rb") as f:
                f.seek(last_size, os.SEEK_SET)
                data = f.read(cur_size - last_size)
            buf += data.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if line.strip():
                    try:
                        state.apply(json.loads(line))
                    except Exception:
                        continue
            last_size = cur_size
        await asyncio.sleep(0.1)


async def main(path: str):
    state = DashboardState()
    console = Console()

    tail_task = asyncio.create_task(tail_events(path, state))
    try:
        with Live(
            _build_layout(state),
            console=console,
            refresh_per_second=10,
            screen=True,
        ) as live:
            while True:
                await asyncio.sleep(0.1)
                live.update(_build_layout(state))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        tail_task.cancel()
        try:
            await tail_task
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=EVENTS_PATH,
                    help=f"events JSONL path (default: {EVENTS_PATH})")
    args = ap.parse_args()
    try:
        asyncio.run(main(args.events))
    except KeyboardInterrupt:
        print("bye", file=sys.stderr)
