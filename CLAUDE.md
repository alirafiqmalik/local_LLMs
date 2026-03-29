# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.
Ground rules: `.claude/ground_rules.md` | Session logs: `.claude/Claude Logs/` | Memory: `.claude/memory/`
**All Claude files live inside the project — never write to `~/.claude/` for project memory.**

## What This Project Is

A unified local LLM inference router — single OpenAI-compatible API (port 8000) across:
- **Ollama** (port 11434): quantized GGUF models via local binary
- **HF Engine** (port 8001): HuggingFace transformers with bitsandbytes quantization

Designed for RTX 3050 6GB GPU. Only one model can live in VRAM at a time. No cloud providers.

## CLI — Single Entry Point

```bash
./llm start [--hf]    # Start router + Ollama (--hf also starts HF engine)
./llm stop            # Stop router and HF engine
./llm restart         # Stop then start
./llm status          # Health check (JSON)
./llm logs [service]  # Tail logs: router | hf | all
./llm models          # List installed Ollama models
./llm pull <tag>      # Pull model: fast | code | general | reasoning | <ollama-tag>
./llm test            # Smoke test all installed models
./llm examples        # Run examples/examples.py
./llm migrate         # Migrate ~/.ollama/models → models/ollama/ (one-time)
./llm hf [--bg|--stop]  # Manage HF engine
./llm help            # Full usage
```

Activate venv: `source .venv/bin/activate`

## Architecture

```
POST /v1/chat/completions
    → classifier.py   (heuristic complexity: TRIVIAL → EXPERT)
    → models.py       (resolve model/provider from routing table)
    → main.py         (prepare_vram: unload other backend first)
    → Ollama | HF Engine
```

**Config:** `.env` at project root — loaded by `router/config.py` (Pydantic BaseSettings).
**Venv:** `.venv/` at project root (not `router/.venv/`).
**Model storage:** `models/ollama/` (Ollama) and `models/huggingface/` (HF).

## `GET /api/models`

Proxies the Ollama model list from `GET {ollama_base_url}/api/tags`. Response JSON:

- **`local_installed`** — array of installed Ollama models, sorted by `id`. Each item has **`id`** (the Ollama tag, e.g. `gemma3:4b`), plus **`size`**, **`modified_at`**, **`digest`** when Ollama returns them. If **`LOCAL_MODELS`** fuzzy-matches that tag, the same object also includes **`vram_gb`**, **`description`**, and **`recommended_for`**.
- **`routing_table`** — `{ trivial, simple, medium_code, medium_general, complex, expert }` string values from `.env` / `router/config.py`.

There is no separate `ollama_models` field; the full install list lives only under **`local_installed`**.

## Virtual Model Tags

| Tag | Routes to |
|-----|-----------|
| `auto` | Complexity-based, always local |
| `local:fast` | gemma3:4b |
| `local:code` | qwen2.5-coder:7b |
| `local:general` | mistral:7b |
| `hf:<model_id>` | HF engine (auto-downloads) |

## Key Design Constraints

- **VRAM sharing:** `OLLAMA_KEEP_ALIVE=0` + `prepare_vram(backend)` in `router/main.py` — do not remove either. Without them, switching backends causes OOM.
- **HF engine single-threaded:** `ThreadPoolExecutor(max_workers=1)` — bitsandbytes is not thread-safe.
- **llama-cpp-python needs CUDA build:** `CMAKE_ARGS="-DGGML_CUDA=on"` — handled by `scripts/start_hf_engine.sh`.

## Routing Table (.env overrides)

```bash
MODEL_TRIVIAL=gemma3:4b
MODEL_SIMPLE=gemma3:4b
MODEL_MEDIUM_CODE=qwen2.5-coder:7b
MODEL_MEDIUM_GENERAL=mistral:7b
MODEL_COMPLEX=mistral:7b
MODEL_EXPERT=mistral:7b
```

## Response Metadata

```python
resp.model_extra["_router"]
# → {"complexity": "medium", "model_used": "qwen2.5-coder:7b", "provider": "ollama", ...}
```

## File Map

```
llm                  Single CLI entry point
scripts/             Internal helper scripts (called by ./llm)
router/              FastAPI app, VRAM coordinator, WebSocket metrics
  main.py            Routes, prepare_vram(), streaming
  config.py          Pydantic settings (reads .env)
  models.py          Model registry + routing resolution
  classifier.py      Heuristic complexity classifier
  metrics.py         System metrics (GPU, CPU, RAM)
engines/hf_engine/   HF inference engine (port 8001)
  main.py            FastAPI, single-threaded executor
  loader.py          HF + llama-cpp-python, VRAM management
dashboard/index.html Dark web UI — real-time WebSocket metrics
docs/                USAGE.md, MODELS.md, Specs.md, hardware_report.md
examples/            Usage examples (examples.py)
.claude/             Ground rules, session logs, memory (all project-local)
  memory/            Persistent memory (MEMORY.md index + *.md files)
```

## Docs

- `docs/USAGE.md` — API usage, curl examples, SDK patterns
- `docs/MODELS.md` — model download guide, HF quantization, GGUF
- `docs/Specs.md` — design decisions, hardware limits, gotchas
