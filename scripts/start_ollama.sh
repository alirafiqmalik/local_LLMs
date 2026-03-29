#!/bin/bash
# ============================================================
# scripts/start_ollama.sh — Start Ollama server with NVIDIA GPU support
# ============================================================
# Usage: bash scripts/start_ollama.sh [--background]
# Canonical interface: ./llm ollama [--bg]
# ============================================================

# BASE_DIR resolves to local_LLMs/ whether called standalone or via llm
BASE_DIR="${BASE_DIR:-"$(cd "$(dirname "$0")/.." && pwd)"}"
OLLAMA_BIN="${BASE_DIR}/ollama_bin/bin/ollama"
OLLAMA_LIB="${BASE_DIR}/ollama_bin/lib/ollama"

export LD_LIBRARY_PATH="${OLLAMA_LIB}:${LD_LIBRARY_PATH}"
export PATH="${BASE_DIR}/ollama_bin/bin:${PATH}"

# Store Ollama models inside local_LLMs/models/ollama/ (portable)
export OLLAMA_MODELS="${BASE_DIR}/models/ollama"
mkdir -p "${OLLAMA_MODELS}"

# Auto-unload model from VRAM after each request — keeps VRAM free for HF engine
export OLLAMA_KEEP_ALIVE=0

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama already running on localhost:11434"
    echo ""
    echo "Installed models:"
    "${OLLAMA_BIN}" list
    echo ""
    echo "GPU status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
    exit 0
fi

echo "Starting Ollama server..."
echo "  Binary: ${OLLAMA_BIN}"
echo "  API:    http://localhost:11434"
echo ""

if [[ "$1" == "--background" || "$1" == "--bg" ]]; then
    mkdir -p "${BASE_DIR}/logs"
    nohup "${OLLAMA_BIN}" serve > "${BASE_DIR}/logs/ollama.log" 2>&1 &
    OLLAMA_PID=$!
    echo "  PID: ${OLLAMA_PID}"

    for i in $(seq 1 30); do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo ""
            echo "✓ Ollama started"
            echo ""
            echo "Installed models:"
            "${OLLAMA_BIN}" list
            echo ""
            echo "GPU status:"
            nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
            echo ""
            echo "  Logs: ${BASE_DIR}/logs/ollama.log"
            exit 0
        fi
        sleep 1
    done
    echo "✗ Ollama failed to start. Check logs/ollama.log"
    exit 1
else
    echo "Running in foreground (Ctrl+C to stop)..."
    echo ""
    "${OLLAMA_BIN}" serve
fi
