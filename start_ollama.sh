#!/bin/bash
# ============================================================
# start_ollama.sh — Start Ollama server with NVIDIA GPU support
# ============================================================
# Usage: bash start_ollama.sh [--background]
#
# This script starts the Ollama server using the local binary
# installation at ~/Desktop/local_LLMs/ollama_bin/
# ============================================================

OLLAMA_DIR="$(cd "$(dirname "$0")" && pwd)"
OLLAMA_BIN="${OLLAMA_DIR}/ollama_bin/bin/ollama"
OLLAMA_LIB="${OLLAMA_DIR}/ollama_bin/lib/ollama"

export LD_LIBRARY_PATH="${OLLAMA_LIB}:${LD_LIBRARY_PATH}"
export PATH="${OLLAMA_DIR}/ollama_bin/bin:${PATH}"

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is already running on localhost:11434"
    echo ""
    echo "Installed models:"
    "${OLLAMA_BIN}" list
    echo ""
    echo "GPU status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
    exit 0
fi

echo "🚀 Starting Ollama server..."
echo "   Binary: ${OLLAMA_BIN}"
echo "   API:    http://localhost:11434"
echo "   OpenAI: http://localhost:11434/v1"
echo ""

if [[ "$1" == "--background" ]]; then
    mkdir -p "${OLLAMA_DIR}/logs"
    nohup "${OLLAMA_BIN}" serve > "${OLLAMA_DIR}/logs/ollama.log" 2>&1 &
    OLLAMA_PID=$!
    echo "   PID:    ${OLLAMA_PID}"
    
    # Wait for server to be ready
    for i in $(seq 1 30); do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo ""
            echo "✅ Ollama server started successfully!"
            echo ""
            echo "Installed models:"
            "${OLLAMA_BIN}" list
            echo ""
            echo "GPU status:"
            nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
            echo ""
            echo "To stop: kill ${OLLAMA_PID}"
            echo "Logs:    ${OLLAMA_DIR}/logs/ollama.log"
            exit 0
        fi
        sleep 1
    done
    echo "❌ Server failed to start. Check ${OLLAMA_DIR}/ollama.log"
    exit 1
else
    echo "Running in foreground (Ctrl+C to stop)..."
    echo ""
    "${OLLAMA_BIN}" serve
fi
