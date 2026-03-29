#!/bin/bash
# ============================================================
# scripts/start_router.sh — Start Ollama + HF Engine + LLM Router
# ============================================================
# Canonical interface: ./llm start [--hf]
# Direct usage:
#   bash scripts/start_router.sh [--bg] [--hf] [--stop]
# ============================================================

set -e

# BASE_DIR resolves to local_LLMs/ whether called standalone or via llm
BASE_DIR="${BASE_DIR:-"$(cd "$(dirname "$0")/.." && pwd)"}"
OLLAMA_BIN="${BASE_DIR}/ollama_bin/bin/ollama"
OLLAMA_LIB="${BASE_DIR}/ollama_bin/lib/ollama"
ROUTER_DIR="${BASE_DIR}/router"
ROUTER_VENV="${BASE_DIR}/.venv"
ROUTER_PID_FILE="${BASE_DIR}/logs/router.pid"
HF_PID_FILE="${BASE_DIR}/logs/hf_engine.pid"

export LD_LIBRARY_PATH="${OLLAMA_LIB}:${LD_LIBRARY_PATH}"
export PATH="${BASE_DIR}/ollama_bin/bin:${PATH}"
export OLLAMA_MODELS="${BASE_DIR}/models/ollama"
export HF_HOME="${BASE_DIR}/models/huggingface"
export TRANSFORMERS_CACHE="${BASE_DIR}/models/huggingface"
export LOCAL_LLMS_DIR="${BASE_DIR}"
export OLLAMA_KEEP_ALIVE=0

mkdir -p "${BASE_DIR}/logs" "${OLLAMA_MODELS}" "${HF_HOME}"

# ── Args ──────────────────────────────────────────────────────────────────────
RUN_BG=false
ENABLE_HF=false
STOP_MODE=false

for arg in "$@"; do
    case "$arg" in
        --bg)   RUN_BG=true ;;
        --hf)   ENABLE_HF=true ;;
        --stop) STOP_MODE=true ;;
        *)
            echo "Unknown flag: $arg"
            echo "Usage: bash scripts/start_router.sh [--bg] [--hf] [--stop]"
            exit 1
            ;;
    esac
done

# ── Stop mode ─────────────────────────────────────────────────────────────────
if [[ "$STOP_MODE" == true ]]; then
    for PID_FILE in "$ROUTER_PID_FILE" "$HF_PID_FILE"; do
        NAME=$(basename "$PID_FILE" .pid)
        if [[ -f "$PID_FILE" ]]; then
            PID=$(cat "$PID_FILE")
            echo "Stopping ${NAME} (PID $PID)..."
            kill "$PID" 2>/dev/null && echo "  ✓ Stopped" || echo "  Already gone"
            rm -f "$PID_FILE"
        else
            echo "${NAME}: no PID file found"
        fi
    done
    exit 0
fi

# ── Step 1: Ollama ────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " LLM Router — Full Stack Startup"
echo " Base: ${BASE_DIR}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama already running"
else
    echo "Starting Ollama..."
    nohup "${OLLAMA_BIN}" serve > "${BASE_DIR}/logs/ollama.log" 2>&1 &
    for i in $(seq 1 30); do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✓ Ollama started  (models: ${OLLAMA_MODELS})"
            break
        fi
        sleep 1
        [[ $i -eq 30 ]] && { echo "✗ Ollama failed. Check logs/ollama.log"; exit 1; }
    done
fi

echo "  Models: $("${OLLAMA_BIN}" list 2>/dev/null | tail -n +2 | awk '{print $1}' | tr '\n' ' ')"
echo ""

# ── Step 2: Router venv + deps ────────────────────────────────────────────────
[[ ! -d "$ROUTER_VENV" ]] && python3 -m venv "$ROUTER_VENV" && echo "✓ Router venv created"
echo "Checking router dependencies..."
"${ROUTER_VENV}/bin/pip" install -q -r "${ROUTER_DIR}/requirements.txt"
echo "✓ Router dependencies ready"
echo ""

# ── Step 3: HF Engine (optional) ──────────────────────────────────────────────
if [[ "$ENABLE_HF" == true ]]; then
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "✓ HF Engine already running on :8001"
    else
        echo "Starting HF Engine..."
        bash "${BASE_DIR}/scripts/start_hf_engine.sh" --bg
    fi
else
    echo "HF Engine disabled (use --hf to enable)."
fi
echo ""

# ── Step 4: GPU status ────────────────────────────────────────────────────────
echo "GPU status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader 2>/dev/null \
    | awk -F', ' '{printf "  %s | VRAM: %s/%s MB | Util: %s%% | %s°C\n",$1,$2,$3,$4,$5}' \
    || echo "  (nvidia-smi not available)"
echo ""

# ── Step 5: Router ────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Router + Dashboard"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Router API : http://localhost:8000/v1/chat/completions"
[[ "$ENABLE_HF" == true ]] && echo "  HF Engine  : http://localhost:8001/v1/chat/completions"
echo "  Dashboard  : http://localhost:8000/"
echo "  Status     : http://localhost:8000/api/status"
echo ""

if [[ "$RUN_BG" == true ]]; then
    nohup "${ROUTER_VENV}/bin/python" "${ROUTER_DIR}/main.py" > "${BASE_DIR}/logs/router.log" 2>&1 &
    ROUTER_PID=$!
    disown $ROUTER_PID
    echo $ROUTER_PID > "$ROUTER_PID_FILE"
    sleep 2
    if kill -0 $ROUTER_PID 2>/dev/null; then
        echo "✓ Router started (PID $ROUTER_PID)"
        echo "  Logs: ${BASE_DIR}/logs/router.log"
        echo "  Stop: ./llm stop"
    else
        echo "✗ Router failed to start. Check logs/router.log"
        exit 1
    fi
else
    echo "Running in foreground (Ctrl+C to stop)..."
    echo ""
    cd "${ROUTER_DIR}"
    exec "${ROUTER_VENV}/bin/python" main.py
fi
