# ============================================================
# LLM Router — Makefile
# Delegates to ./llm CLI. Kept for muscle-memory compatibility.
# Canonical interface: ./llm help
# ============================================================

.PHONY: start stop restart status logs test examples \
        hf-start hf-stop ollama-start \
        pull-fast pull-code pull-general pull-reasoning \
        models migrate help

# ── Start / Stop ─────────────────────────────────────────────

start:         ## Start everything (background)
	./llm start

stop:          ## Stop router + HF engine
	./llm stop

restart: stop start  ## Restart all services

# ── Individual services ───────────────────────────────────────

ollama-start:  ## Start Ollama only
	./llm ollama --bg

hf-start:      ## Start HF engine only (port 8001)
	./llm hf --bg

hf-stop:       ## Stop HF engine only
	./llm hf --stop

# ── Status & Logs ─────────────────────────────────────────────

status:        ## Show router + Ollama + HF engine health
	./llm status

logs:          ## Tail router + HF engine logs
	./llm logs

# ── Testing ───────────────────────────────────────────────────

test:          ## Test all installed Ollama models
	./llm test

examples:      ## Run usage examples (router must be running)
	./llm examples

# ── Model management ──────────────────────────────────────────

pull-fast:     ## Pull fast triage model (phi3.5:3.8b, 2.3 GB)
	./llm pull fast

pull-code:     ## Pull code model (qwen2.5-coder:7b, 4.7 GB)
	./llm pull code

pull-general:  ## Pull general model (mistral:7b, 4.4 GB)
	./llm pull general

pull-reasoning: ## Pull reasoning model (deepseek-r1:7b, 4.9 GB)
	./llm pull reasoning

models:        ## List installed Ollama models
	./llm models

migrate:       ## Migrate ~/.ollama/models → models/ollama/ (one-time)
	./llm migrate

# ── Help ──────────────────────────────────────────────────────

help:          ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile \
	    | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'
	@echo ""
	@echo "  Full CLI:  ./llm help"

.DEFAULT_GOAL := help
