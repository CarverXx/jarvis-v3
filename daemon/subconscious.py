"""
Subconscious — fast direct-to-SGLang chat layer. Answers 80% of daily
chitchat itself, escalates complex tasks to Hermes via the invoke_hermes
tool call. Maintains its own sliding-window message history so the SGLang
side is effectively stateless.

Shape of a turn:
    turn(user_text)
      → append {"role":"user","content":user_text} to history
      → call SGLang /v1/chat/completions with tools=[invoke_hermes]
      → if response.tool_calls:
            for each tc, run executor (blocks — loop plays waiting audio)
            append {"role":"assistant",content:None,tool_calls:[...]}
            append {"role":"tool",tool_call_id:..,content:tool_result}
            loop back for summarisation
        else:
            append {"role":"assistant","content":reply}
            return reply
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

from daemon import backend_client
from daemon.tools import ALL_TOOLS, TOOL_EXECUTORS
from daemon import events

logger = logging.getLogger("subconscious")

MAX_TOOL_HOPS = 2   # cap subconscious↔tool round-trips to avoid loops


class Subconscious:
    def __init__(self, session_id: str):
        self.session_id = session_id
        # History never includes the system prompt; we prepend it on every request.
        self.history: list[dict[str, Any]] = []

    # ---- history management ------------------------------------------------
    def _truncate(self):
        # Keep at most N user turns + their assistant/tool replies.
        # Count user messages; slice from the oldest user turn exceeding N.
        user_idxs = [i for i, m in enumerate(self.history) if m["role"] == "user"]
        if len(user_idxs) > cfg.SUBCONSCIOUS_HISTORY_N:
            cut = user_idxs[-cfg.SUBCONSCIOUS_HISTORY_N]
            self.history = self.history[cut:]

    def _build_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": cfg.SUBCONSCIOUS_SYSTEM_PROMPT},
            *self.history,
        ]

    # ---- main entry --------------------------------------------------------
    async def chat(self, user_text: str, *, waiting_player=None) -> str:
        """Returns the final spoken reply (after any tool-call round-trip)."""
        self.history.append({"role": "user", "content": user_text})
        self._truncate()
        events.log("sub_start", user_text=user_text)
        turn_t0 = time.monotonic()

        for hop in range(MAX_TOOL_HOPS + 1):
            t0 = time.monotonic()
            raw = await backend_client.subconscious_chat(
                self._build_messages(),
                tools=ALL_TOOLS,
                event_hop=hop,   # for chunk emission in stream mode
            )
            elapsed = time.monotonic() - t0
            msg = raw["choices"][0]["message"]
            content = (msg.get("content") or "").strip() or None
            tool_calls = msg.get("tool_calls") or []
            logger.info(
                "hop=%d subconscious %.2fs content=%r tools=%d",
                hop, elapsed,
                (content[:80] if content else None),
                len(tool_calls),
            )

            if not tool_calls:
                # Final answer path
                final = content or "(我没听清楚, 请再说一遍)"
                self.history.append({"role": "assistant", "content": final})
                events.log("sub_done", content=final, tool_called=False,
                           elapsed_s=round(time.monotonic() - turn_t0, 3))
                return final

            if hop >= MAX_TOOL_HOPS:
                # Safety net — refuse to loop; ask Hermes to finalise.
                logger.warning("max tool hops hit; returning last tool output")
                # Find last tool output if any to say something useful
                last_tool = next(
                    (m for m in reversed(self.history) if m["role"] == "tool"),
                    None,
                )
                fallback = (
                    last_tool["content"] if last_tool
                    else "我在处理, 但是有点问题, 请再试一次。"
                )
                self.history.append({"role": "assistant", "content": fallback})
                events.log("sub_done", content=fallback, tool_called=True,
                           elapsed_s=round(time.monotonic() - turn_t0, 3),
                           max_hops_hit=True)
                return fallback

            # Record assistant turn that includes the tool_calls so the next
            # request keeps a consistent OpenAI-format trace.
            self.history.append({
                "role": "assistant",
                "content": content,   # may be None when only tool_calls present
                "tool_calls": tool_calls,
            })

            # Execute each tool call and record its result.
            for tc in tool_calls:
                tc_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                fn = tc["function"]["name"]
                raw_args = tc["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {"task": str(raw_args)}
                events.log("sub_tool", tool=fn, args=args, tool_call_id=tc_id)

                executor = TOOL_EXECUTORS.get(fn)
                if executor is None:
                    tool_result = f"(未知工具: {fn})"
                    logger.warning("unknown tool requested: %s", fn)
                else:
                    try:
                        tool_result = await executor(
                            **args,
                            session_id=self.session_id,
                            waiting_player=waiting_player,
                        )
                    except TypeError:
                        # Executor signature mismatch fallback
                        tool_result = await executor(**args)
                    except Exception as e:
                        logger.exception("tool %s failed", fn)
                        tool_result = f"(工具执行失败: {e})"

                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_result,
                })

            # Loop back: let the subconscious summarise the tool output.

        # Unreachable
        return "(潜意识处理异常)"
