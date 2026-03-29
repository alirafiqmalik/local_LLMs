# Project Specs — Local LLM Stack

Non-obvious design decisions, constraints, and lessons learned.
Not covered by USAGE.md, MODELS.md, or inline code comments.

---

## Hardware Constraints

| Component | Spec | Impact |
|-----------|------|--------|
| GPU | RTX 3050 6GB VRAM | Hard ceiling — only one 7B model in VRAM at a time |
| RAM | 16GB + 16GB swap | CPU-offload possible but slow; 30B+ models are impractical |
| CPU | i5-13500H 12c/16t | Handles CPU-spill for mid-size HF models |
| iGPU | Intel Iris Xe (shared RAM) | Not used — deferred |

**6GB is the hard limit.** A quantized 7B model takes ~4–5GB. Two 7B models simultaneously = OOM crash.

---

## Critical: VRAM Sharing Architecture

Both Ollama and the HF engine share the same 6GB GPU. The mechanism that keeps this safe:

1. `OLLAMA_KEEP_ALIVE=0` — Ollama auto-unloads its model from VRAM after every request (set in all start scripts and the `llm` CLI)
2. `prepare_vram(backend)` in `router/main.py` — before each inference call, explicitly tells the *other* engine to unload:
   - If routing to Ollama → calls `POST http://localhost:8001/v1/models/unload` (HF engine)
   - If routing to HF engine → calls Ollama's `/api/ps` then `/api/chat` with `keep_alive=0`

**Do not remove `OLLAMA_KEEP_ALIVE=0` or `prepare_vram()`.** Without them, the second request after a model switch will OOM.

---

## Why Two Separate Engine Processes

| Engine | Port | Best for |
|--------|------|---------|
| Ollama | 11434 | Pre-quantized GGUF models, fast cold start, CLI management |
| HF Engine | 8001 | Any HuggingFace Hub model, bitsandbytes 4-bit, GGUF files |

They cannot be merged — Ollama is a standalone binary, not a Python library. The router at port 8000 abstracts both behind one OpenAI-compatible API.

---

## Venv Location

**Venv is at `.venv/` (project root), NOT `router/.venv/`.**

`scripts/start_router.sh` uses `"${BASE_DIR}/.venv/bin/python"` with an absolute path to `router/main.py` — it does NOT `cd` into `router/` before launching the background process.

If you recreate the venv, create it at `local_LLMs/.venv`:
```bash
python3 -m venv .venv
.venv/bin/pip install -r router/requirements.txt
```

---

## Router Startup — What Was Broken and Why

Original bug: router was started manually from `router/` dir as `.venv/bin/python main.py`. This caused:
- No PID file written → `--stop` didn't work
- Not `disown`-ed → could die on session end in some shells

Fixed in `scripts/start_router.sh --bg` (and `./llm start`):
- Uses absolute path: `nohup "${ROUTER_VENV}/bin/python" "${ROUTER_DIR}/main.py"`
- `disown $ROUTER_PID` after fork to fully detach from shell session
- PID written to `logs/router.pid` immediately

**Always use `./llm start` to start. Never start `main.py` manually.**

---

## HF Engine: Thread Safety

`engines/hf_engine/main.py` uses `ThreadPoolExecutor(max_workers=1)`. This is intentional — bitsandbytes and llama-cpp-python are not thread-safe for inference. Only one request runs at a time. Do not increase `max_workers`.

---

## llama-cpp-python Must Be Built with CUDA

Install with:
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

A standard `pip install llama-cpp-python` installs CPU-only and silently falls back to CPU inference (~10x slower). The `scripts/start_hf_engine.sh` script handles this automatically.

---

## Models That Do NOT Fit on This Hardware

| Model | Why it doesn't fit |
|-------|--------------------|
| GLM-4.5-Air (smallest) | ~40GB IQ1_M quantization — exceeds 16GB RAM + 16GB swap |
| GLM-4.5 (full) | >100GB |
| Any 70B model | ~40GB in 4-bit |
| Any 30B model | ~18GB in 4-bit — possible CPU-only but impractical |

For GLM-family: use `glm4:9b-chat-q4_0` (~5.5GB) via Ollama instead.

---

## HuggingFace Token

Add to `.env` for gated models (Llama-3, Phi-4, Gemma-3):
```
HF_TOKEN=hf_xxxx
```

Without it: public models work fine, gated models will fail with 403.

---

## Model Storage Layout

```
models/
├── ollama/          ← OLLAMA_MODELS env var points here
└── huggingface/     ← HF_HOME + TRANSFORMERS_CACHE point here
```

Ollama looks for models via `OLLAMA_MODELS`. If that env var is not set (e.g., running Ollama outside of `./llm`), it falls back to `~/.ollama/models` and won't find the models.

---

## Portable Paths

All Python files derive `BASE_DIR` from `Path(__file__).resolve().parent...` — the project can be moved to any directory without code changes. All shell scripts use `BASE_DIR="${BASE_DIR:-"$(cd "$(dirname "$0")/.." && pwd)"}"` with the same principle.

The only hardcoded location is the Ollama binary at `ollama_bin/bin/ollama`.

---

## Monorepo Layout

```
scripts/          Internal shell scripts (called by ./llm)
docs/             USAGE.md, MODELS.md, Specs.md, hardware_report.md
examples/         Usage examples (examples.py)
router/           FastAPI router service (port 8000)
engines/hf_engine HF inference engine (port 8001)
dashboard/        Web UI (served at /)
models/           Model storage (gitignored)
logs/             Runtime logs and PID files
ollama_bin/       Local Ollama binary (gitignored)
```
