#!/bin/bash
# ============================================================
# start_router.sh — Start Ollama + LLM Router + Dashboard
# ============================================================
# Usage:
#   bash start_router.sh           # start everything (foreground router)
#   bash start_router.sh --bg      # start router in background
#   bash start_router.sh --stop    # stop router process
# ============================================================

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
OLLAMA_BIN="${DIR}/ollama_bin/bin/ollama"
OLLAMA_LIB="${DIR}/ollama_bin/lib/ollama"
ROUTER_DIR="${DIR}/router"
VENV_DIR="${DIR}/router/.venv"
ROUTER_PID_FILE="${DIR}/logs/router.pid"

export LD_LIBRARY_PATH="${OLLAMA_LIB}:${LD_LIBRARY_PATH}"
export PATH="${DIR}/ollama_bin/bin:${PATH}"

# ── Stop mode ──────────────────────────────────────────────────────────────────
if [[ "$1" == "--stop" ]]; then
    if [[ -f "$ROUTER_PID_FILE" ]]; then
        PID=$(cat "$ROUTER_PID_FILE")
        echo "Stopping LLM Router (PID $PID)..."
        kill "$PID" 2>/dev/null && echo "✓ Stopped" || echo "Process already gone"
        rm -f "$ROUTER_PID_FILE"
    else
        echo "No router PID file found."
    fi
    exit 0
fi

# ── Step 1: Start Ollama if not running ────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " LLM Router — Startup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama already running"
else
    echo "Starting Ollama server..."
    nohup "${OLLAMA_BIN}" serve > "${DIR}/ollama.log" 2>&1 &
    echo "  Waiting for Ollama..."
    for i in $(seq 1 30); do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✓ Ollama started"
            break
        fi
        sleep 1
        if [[ $i -eq 30 ]]; then
            echo "✗ Ollama failed to start. Check ollama.log"
            exit 1
        fi
    done
fi

echo ""
echo "Installed models:"
"${OLLAMA_BIN}" list 2>/dev/null || echo "  (could not list models)"
echo ""

# ── Step 2: Set up Python venv ─────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating Python venv..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Venv created at $VENV_DIR"
fi

echo "Installing/checking Python dependencies..."
"${VENV_DIR}/bin/pip" install -q -r "${ROUTER_DIR}/requirements.txt"
echo "✓ Dependencies ready"
echo ""

# ── Step 3: GPU status ─────────────────────────────────────────────────────────
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader 2>/dev/null \
    | awk -F', ' '{printf "  %s | VRAM: %s/%s MB | Util: %s%% | Temp: %s°C\n",$1,$2,$3,$4,$5}' \
    || echo "  (nvidia-smi not available)"
echo ""

# ── Step 4: Start router ───────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Starting LLM Router API + Dashboard"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  API:       http://localhost:8000/v1/chat/completions"
echo "  Dashboard: http://localhost:8000/"
echo "  Status:    http://localhost:8000/api/status"
echo "  Models:    http://localhost:8000/api/models"
echo ""

cd "${ROUTER_DIR}"

if [[ "$1" == "--bg" ]]; then
    mkdir -p "${DIR}/logs"
    nohup "${VENV_DIR}/bin/python" main.py > "${DIR}/logs/router.log" 2>&1 &
    ROUTER_PID=$!
    echo $ROUTER_PID > "$ROUTER_PID_FILE"
    sleep 2
    if kill -0 $ROUTER_PID 2>/dev/null; then
        echo "✓ Router started (PID $ROUTER_PID)"
        echo "  Logs: ${DIR}/logs/router.log"
        echo "  Stop: bash start_router.sh --stop"
    else
        echo "✗ Router failed to start. Check router.log"
        exit 1
    fi
else
    echo "Running in foreground (Ctrl+C to stop)..."
    echo ""
    exec "${VENV_DIR}/bin/python" main.py
fi
