#!/bin/bash
# ============================================================
# scripts/start_hf_engine.sh — Start the HuggingFace inference engine
# ============================================================
# Usage:
#   bash scripts/start_hf_engine.sh          # foreground
#   bash scripts/start_hf_engine.sh --bg     # background
#   bash scripts/start_hf_engine.sh --stop   # stop
# Canonical interface: ./llm hf [--bg|--stop]
# ============================================================

set -e

# BASE_DIR resolves to local_LLMs/ whether called standalone or via llm
BASE_DIR="${BASE_DIR:-"$(cd "$(dirname "$0")/.." && pwd)"}"
ENGINE_DIR="${BASE_DIR}/engines/hf_engine"
VENV_DIR="${ENGINE_DIR}/.venv"
PID_FILE="${BASE_DIR}/logs/hf_engine.pid"
LOG_FILE="${BASE_DIR}/logs/hf_engine.log"
HF_MODELS_DIR="${BASE_DIR}/models/huggingface"
HF_ENGINE_PORT="${HF_ENGINE_PORT:-8001}"

mkdir -p "${BASE_DIR}/logs" "${HF_MODELS_DIR}"

# ── Stop mode ─────────────────────────────────────────────────────────────────
if [[ "$1" == "--stop" ]]; then
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        echo "Stopping HF Engine (PID $PID)..."
        kill "$PID" 2>/dev/null && echo "✓ Stopped" || echo "Process already gone"
        rm -f "$PID_FILE"
    else
        echo "No HF engine PID file found."
    fi
    exit 0
fi

# ── Check if already running ──────────────────────────────────────────────────
if curl -s "http://localhost:${HF_ENGINE_PORT}/health" > /dev/null 2>&1; then
    echo "✓ HF Engine already running on port ${HF_ENGINE_PORT}"
    curl -s "http://localhost:${HF_ENGINE_PORT}/health" | python3 -m json.tool 2>/dev/null
    exit 0
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " HF Engine — Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Base dir  : ${BASE_DIR}"
echo "  Models dir: ${HF_MODELS_DIR}"
echo "  Port      : ${HF_ENGINE_PORT}"
echo ""

# ── Create venv if needed ─────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating Python venv..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Venv created"
fi

# ── Install / check dependencies ──────────────────────────────────────────────
echo "Installing/checking Python dependencies..."
"${VENV_DIR}/bin/pip" install -q -r "${ENGINE_DIR}/requirements.txt"

# Install llama-cpp-python with CUDA support (separate — needs build flags)
if ! "${VENV_DIR}/bin/python" -c "import llama_cpp" 2>/dev/null; then
    echo "Installing llama-cpp-python with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=on" \
        "${VENV_DIR}/bin/pip" install -q llama-cpp-python --force-reinstall --no-cache-dir \
        2>&1 | tail -3
fi
echo "✓ Dependencies ready"
echo ""

# ── Export env for the engine ─────────────────────────────────────────────────
export HF_ENGINE_PORT
export HF_HOME="${HF_MODELS_DIR}"
export TRANSFORMERS_CACHE="${HF_MODELS_DIR}"
export LOCAL_LLMS_DIR="${BASE_DIR}"

# Load HF_TOKEN from .env if not already set
if [[ -z "$HF_TOKEN" && -f "${BASE_DIR}/.env" ]]; then
    HF_TOKEN_VAL=$(grep -E "^HF_TOKEN=" "${BASE_DIR}/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")
    if [[ -n "$HF_TOKEN_VAL" ]]; then
        export HF_TOKEN="$HF_TOKEN_VAL"
        echo "  HF_TOKEN: ✓ (loaded from .env)"
    else
        echo "  HF_TOKEN: not set (add to .env for gated models)"
    fi
fi
echo ""

# ── Start engine ──────────────────────────────────────────────────────────────
cd "${ENGINE_DIR}"

if [[ "$1" == "--bg" ]]; then
    echo "Starting HF Engine in background..."
    nohup "${VENV_DIR}/bin/python" main.py > "${LOG_FILE}" 2>&1 &
    HF_PID=$!
    echo $HF_PID > "$PID_FILE"

    for i in $(seq 1 30); do
        if curl -s "http://localhost:${HF_ENGINE_PORT}/health" > /dev/null 2>&1; then
            echo "✓ HF Engine started (PID $HF_PID)"
            echo "  API : http://localhost:${HF_ENGINE_PORT}/v1/chat/completions"
            echo "  Logs: ${LOG_FILE}"
            echo "  Stop: ./llm hf --stop"
            exit 0
        fi
        sleep 1
    done
    echo "✗ HF Engine failed to start. Check ${LOG_FILE}"
    exit 1
else
    echo "Running in foreground (Ctrl+C to stop)..."
    echo "  API: http://localhost:${HF_ENGINE_PORT}/v1/chat/completions"
    echo ""
    exec "${VENV_DIR}/bin/python" main.py
fi
