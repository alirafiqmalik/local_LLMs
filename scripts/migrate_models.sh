#!/bin/bash
# ============================================================
# scripts/migrate_models.sh — Move Ollama models into local_LLMs/
# ============================================================
# Run once to relocate ~/.ollama/models → local_LLMs/models/ollama/
# Canonical interface: ./llm migrate
# Direct usage: bash scripts/migrate_models.sh
# ============================================================

BASE_DIR="${BASE_DIR:-"$(cd "$(dirname "$0")/.." && pwd)"}"
SRC="${HOME}/.ollama/models"
DST="${BASE_DIR}/models/ollama"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Ollama Model Migration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  From: ${SRC}"
echo "  To  : ${DST}"
echo ""

mkdir -p "${DST}"

if [[ ! -d "$SRC" ]]; then
    echo "✗ Source not found: ${SRC}"
    echo "  Nothing to migrate — models may already be in the right place."
    exit 0
fi

SRC_SIZE=$(du -sh "${SRC}" 2>/dev/null | cut -f1)
echo "  Source size: ${SRC_SIZE}"
echo ""
read -p "Copy models to ${DST}? (y/N) " CONFIRM
[[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]] && echo "Aborted." && exit 0

echo ""
echo "Copying (this may take a few minutes)..."
cp -r "${SRC}/." "${DST}/"
echo ""
echo "✓ Migration complete."
echo ""
echo "Next steps:"
echo "  1. Verify: ./llm models"
echo "  2. Once verified, free space with: rm -rf ${SRC}"
echo ""
echo "OLLAMA_MODELS is set automatically by ./llm and scripts/start_router.sh"
