# Local LLM Router

A small **OpenAI-compatible** HTTP API on **port 8000** in front of **Ollama** (GGUF models) and an optional **Hugging Face engine** (port 8001). One familiar endpoint and a single CLI (`./llm`) instead of wiring ports, clients, and memory behavior yourself every time.

## Why this exists

Running models locally often means juggling Ollama, a separate HF stack, different ports and APIs, and awkward VRAM behavior when you switch apps or backends. Many tools and scripts already speak the OpenAI chat API. On a **small GPU** you cannot afford to leave large models loaded while you bounce between setups.

This router **maps or chooses models** (including an **`auto`** path that guesses how demanding a prompt is), forwards to the right backend, and **coordinates VRAM** so switching backends or model sizes is less likely to blow memory. Everything stays **private and offline-capable** once models are pulled.

## What you get

- **`POST /v1/chat/completions`** — same shape clients expect from OpenAI-style APIs.
- **Virtual tags** like `auto`, `local:fast`, `local:code`, `hf:<model_id>` — see `docs/MODELS.md` and `.env` / `router/config.py` for the routing table.
- **Optional dashboard** — `http://localhost:8000/` when the router is up.
- **`./llm`** — start/stop services, logs, model pulls, smoke tests, and helpers documented in `./llm help`.

## Quick start

```bash
./llm start              # router + Ollama
./llm start --hf         # also start the HF engine
./llm status             # health JSON
```

Point any OpenAI-compatible client at `http://localhost:8000/v1` (API key can be any string, e.g. `local`). Example with `model="auto"` is in `docs/USAGE.md`.

Python work on the router itself uses the project venv:

```bash
source .venv/bin/activate
```

## Docs

| Doc | Contents |
|-----|----------|
| [docs/USAGE.md](docs/USAGE.md) | API usage, curl, SDK patterns |
| [docs/MODELS.md](docs/MODELS.md) | Pulling models, quantization notes |
| [docs/Specs.md](docs/Specs.md) | Design choices, hardware limits |
| [CLAUDE.md](CLAUDE.md) | Maintainer-oriented architecture map |

## Roadmap (work in progress)

The current design targets **one modest GPU** (e.g. ~6GB VRAM) and **one heavy model resident at a time**, with explicit steps to avoid OOM when changing backends. Longer term the direction includes **smarter scheduling under concurrency**, support for **multiple small GPUs or Apple Silicon**, and **tighter, analyzable memory bounds** so average latency and throughput improve without sacrificing safety. That scheduling and formal guarantees are **not** fully realized in this repo yet; what ships today is practical routing and VRAM discipline for daily local use.
