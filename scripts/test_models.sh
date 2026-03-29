#!/bin/bash
# ============================================================
# scripts/test_models.sh — Test all installed Ollama models
# ============================================================
# Sends a test prompt to each model and reports response time.
# Canonical interface: ./llm test
# Direct usage: bash scripts/test_models.sh
# ============================================================

OLLAMA_URL="http://localhost:11434"

if ! curl -s "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    echo "✗ Ollama not running. Start with: ./llm start"
    exit 1
fi

echo "========================================="
echo "  Ollama Model Test Suite"
echo "  $(date)"
echo "========================================="
echo ""

MODELS=$(curl -s "${OLLAMA_URL}/api/tags" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    print(m['name'])
" 2>/dev/null)

if [ -z "$MODELS" ]; then
    echo "✗ No models installed."
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  N/A"
echo ""

PASS=0
FAIL=0

for MODEL in $MODELS; do
    echo "-----------------------------------------"
    echo "Testing: ${MODEL}"
    echo "-----------------------------------------"

    if echo "$MODEL" | grep -qi "embed"; then
        echo "  Embedding model — testing via /api/embed"
        RESULT=$(curl -s "${OLLAMA_URL}/api/embed" \
            -d "{\"model\": \"${MODEL}\", \"input\": \"Hello world\"}" 2>/dev/null)

        if echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
embeddings = data.get('embeddings', [])
if embeddings and len(embeddings[0]) > 0:
    print(f'  ✓ Embedding dimension: {len(embeddings[0])}')
else:
    print('  ✗ No embeddings returned')
    sys.exit(1)
" 2>/dev/null; then
            PASS=$((PASS + 1))
        else
            echo "  ✗ Failed"
            FAIL=$((FAIL + 1))
        fi
        echo ""
        continue
    fi

    START_TIME=$(date +%s%N)
    RESULT=$(curl -s "${OLLAMA_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"${MODEL}\",
            \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer with just the number.\"}],
            \"stream\": false,
            \"options\": {\"num_predict\": 10}
        }" 2>/dev/null)
    END_TIME=$(date +%s%N)
    ELAPSED=$(( (END_TIME - START_TIME) / 1000000 ))

    RESPONSE=$(echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
choices = data.get('choices', [])
if choices:
    print(choices[0].get('message', {}).get('content', '').strip())
else:
    print('ERROR: ' + json.dumps(data))
" 2>/dev/null)

    USAGE=$(echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
u = data.get('usage', {})
print(f'prompt={u.get(\"prompt_tokens\",\"?\")}, completion={u.get(\"completion_tokens\",\"?\")}, total={u.get(\"total_tokens\",\"?\")}')
" 2>/dev/null)

    if [ -n "$RESPONSE" ] && ! echo "$RESPONSE" | grep -q "ERROR"; then
        echo "  ✓ Response: ${RESPONSE}"
        echo "    Time: ${ELAPSED}ms  |  ${USAGE}"
        PASS=$((PASS + 1))
    else
        echo "  ✗ Failed: ${RESPONSE}"
        FAIL=$((FAIL + 1))
    fi
    echo ""
done

echo "========================================="
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "========================================="
echo ""
echo "GPU Memory After Tests:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "  N/A"
