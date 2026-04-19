#!/bin/bash
# sglang-launcher.sh — boots SGLang openai-compatible server from
# /opt/sglang/config.json (override via $SGLANG_CONFIG). Invoked by
# /etc/systemd/system/sglang.service ExecStart.
#
# Install:
#   sudo mkdir -p /opt/sglang
#   sudo cp scripts/sglang-launcher.sh /opt/sglang/
#   sudo chmod +x /opt/sglang/sglang-launcher.sh
#   sudo cp scripts/sglang-config.example.json /opt/sglang/config.json  # then edit

set -euo pipefail

CONFIG=${SGLANG_CONFIG:-/opt/sglang/config.json}
MODELS_DIR=${HOME}/models
VENV=/opt/sglang-env

if [ ! -f "$CONFIG" ]; then
    echo "FATAL: $CONFIG not found" >&2
    exit 1
fi

MODEL_ID=$(jq -r '.model_id // empty' "$CONFIG")
if [ -z "$MODEL_ID" ]; then
    echo "FATAL: .model_id missing in $CONFIG" >&2
    exit 1
fi
MODEL_PATH="$MODELS_DIR/$MODEL_ID"
if [ ! -d "$MODEL_PATH" ]; then
    echo "FATAL: model dir $MODEL_PATH does not exist" >&2
    exit 1
fi

CONTEXT_LEN=$(jq -r '.context_length // 131072' "$CONFIG")
MEM_FRAC=$(jq -r '.mem_fraction_static // 0.78' "$CONFIG")
TP_SIZE=$(jq -r '.tp_size // 1' "$CONFIG")
HOST=$(jq -r '.host // "0.0.0.0"' "$CONFIG")
PORT=$(jq -r '.port // 8000' "$CONFIG")
TOOL_PARSER=$(jq -r '.tool_call_parser // empty' "$CONFIG")
REASONING_PARSER=$(jq -r '.reasoning_parser // empty' "$CONFIG")
TRUST_REMOTE=$(jq -r '.trust_remote_code // true' "$CONFIG")
API_KEY=$(jq -r '.api_key // empty' "$CONFIG")
CHAT_TEMPLATE=$(jq -r '.chat_template // empty' "$CONFIG")
mapfile -t EXTRA_ARGS < <(jq -r '.extra_args[]?' "$CONFIG")

ARGS=(
    --model-path "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --tp-size "$TP_SIZE"
    --mem-fraction-static "$MEM_FRAC"
    --context-length "$CONTEXT_LEN"
)

[ "$TRUST_REMOTE" = "true" ] && ARGS+=(--trust-remote-code)
[ -n "$TOOL_PARSER" ] && ARGS+=(--tool-call-parser "$TOOL_PARSER")
[ -n "$REASONING_PARSER" ] && ARGS+=(--reasoning-parser "$REASONING_PARSER")
[ -n "$API_KEY" ] && ARGS+=(--api-key "$API_KEY")
[ -n "$CHAT_TEMPLATE" ] && ARGS+=(--chat-template "$CHAT_TEMPLATE")

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    ARGS+=("${EXTRA_ARGS[@]}")
fi

# Avoid leaking api_key to logs — print a redacted command
echo "[sglang-launcher] starting model=$MODEL_ID context=$CONTEXT_LEN mem=$MEM_FRAC"
REDACTED=()
skip_next=false
for arg in "${ARGS[@]}"; do
    if $skip_next; then REDACTED+=("***REDACTED***"); skip_next=false; continue; fi
    if [ "$arg" = "--api-key" ]; then REDACTED+=("$arg"); skip_next=true; continue; fi
    REDACTED+=("$arg")
done
echo "[sglang-launcher] cmd: python -m sglang.launch_server ${REDACTED[*]}"

exec "$VENV/bin/python" -m sglang.launch_server "${ARGS[@]}"
