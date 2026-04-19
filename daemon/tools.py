"""
Tools exposed to the subconscious — currently only `invoke_hermes`.

The subconscious (SGLang Qwen3.6) calls these via OpenAI-format
tool_calls when it decides a turn needs deep agentic processing.
Results are routed back into the conversation as {role: "tool"} messages
so the model can summarise in natural Chinese for the TTS layer.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

from daemon import backend_client
from daemon import events

logger = logging.getLogger("tools")


# OpenAI Chat Completions `tools=[...]` item format.
INVOKE_HERMES_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "invoke_hermes",
        "description": (
            "委托给主意识 Hermes Agent 完成需要深度推理 / 多工具协作 / 访问文件系统 / "
            "执行命令 / 实时联网搜索 / 操作家庭 NAS 或 Raphael 知识库 / 代码生成 / "
            "数据分析 / 长任务（预期 >3 秒）。Hermes 返回一段简短结果文本。"
        ),
        # timeout_s removed 2026-04-19 — LLM kept picking 30s even with default=90.
        # Daemon side enforces 90s via execute_invoke_hermes(timeout_s=90).
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "中文描述任务。可直接转述用户原话或重写得更明确。"
                },
            },
            "required": ["task"],
        },
    },
}


async def execute_invoke_hermes(
    task: str,
    *,
    session_id: str,
    timeout_s: int = 90,   # Was 30 — web-search weather queries routinely hit 30s,
                           # see 2026-04-19 log at 13:38:15 '查询今天上海的天气' ReadTimeout
    waiting_player=None,
) -> str:
    """Run the Hermes tool-call.

    Starts the heartbeat waiting-loop audio (via `waiting_player`), delegates
    to the hermes-shim HTTP service (:8004) which in turn runs the `hermes`
    CLI subprocess, then stops the loop. Returns the Hermes text reply as
    the tool's string result, to be fed back into the subconscious context
    for final summarisation.
    """
    loop_handle = None
    if waiting_player is not None:
        try:
            loop_handle = await waiting_player.start_loop(
                cfg.WAITING_BEEP_WAV,
                interval_s=cfg.WAITING_BEEP_INTERVAL_S,
            )
        except Exception:
            logger.exception("waiting-loop start failed (continuing without audio cue)")

    logger.info("invoke_hermes · delegating task=%r timeout=%ds", task[:120], timeout_s)
    start_t = asyncio.get_event_loop().time()
    events.log("hermes_start", task=task, timeout_s=timeout_s)

    # Tick task — fires every WAITING_BEEP_INTERVAL_S so the TUI can show
    # progress animation while Hermes is in-flight. Cancelled in the finally.
    async def _tick():
        try:
            while True:
                await asyncio.sleep(cfg.WAITING_BEEP_INTERVAL_S)
                events.log("hermes_tick",
                           elapsed_s=round(asyncio.get_event_loop().time() - start_t, 1))
        except asyncio.CancelledError:
            return

    tick_task = asyncio.create_task(_tick(), name="hermes-tick")

    ok = True
    try:
        reply = await backend_client.chat(
            task,
            session_id=session_id,
            timeout_s=timeout_s,
        )
    except Exception as e:
        logger.exception("invoke_hermes failed")
        reply = f"(主意识处理失败: {e})"
        ok = False
    finally:
        tick_task.cancel()
        try:
            await tick_task
        except Exception:
            pass
        if loop_handle is not None and waiting_player is not None:
            try:
                await waiting_player.stop_loop(loop_handle)
            except Exception:
                logger.exception("waiting-loop stop failed")

    elapsed = asyncio.get_event_loop().time() - start_t
    events.log("hermes_done", reply=reply, ok=ok, elapsed_s=round(elapsed, 2))
    logger.info("invoke_hermes · reply %r", reply[:120] if reply else "")
    return reply or "(主意识无回应)"


TOOL_EXECUTORS = {
    "invoke_hermes": execute_invoke_hermes,
}


ALL_TOOLS = [INVOKE_HERMES_SCHEMA]
