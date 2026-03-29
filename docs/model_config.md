# LLM Router — Configuration & Model Reference

## API Endpoints

| Endpoint | URL | Purpose |
|----------|-----|---------|
| **LLM Router** | `http://localhost:8000/v1` | Main entry point — smart routing |
| **Dashboard** | `http://localhost:8000/` | Live resource monitor |
| **HF Engine** | `http://localhost:8001/v1` | HuggingFace inference (direct) |
| **Ollama** | `http://localhost:11434/v1` | Ollama direct (bypass router) |

Always call the **router at :8000** — it handles everything.

---

## Routing Table

| Complexity | Trigger | Model |
|-----------|---------|-------|
| `trivial` | greetings, one-liners | `gemma3:4b` |
| `simple` | short Q&A | `gemma3:4b` |
| `medium/code` | coding tasks, debugging | `qwen2.5-coder:7b-instruct-q4_K_M` |
| `medium/general` | explanations, planning | `mistral:7b-instruct-q4_K_M` |
| `complex` | refactoring, algorithms | `mistral:7b-instruct-q4_K_M` |
| `expert` | architecture, system design | cloud API (if key set) → else best local |

---

## Installed Ollama Models

| Model | Role | VRAM | Speed |
|-------|------|------|-------|
| `gemma3:4b` | Fast triage, chat | 3.3 GB | ★★★★★ |
| `qwen2.5-coder:7b-instruct-q4_K_M` | Code gen, debug | 4.7 GB | ★★★ |
| `mistral:7b-instruct-q4_K_M` | Reasoning, analysis | 4.4 GB | ★★★ |
| `nomic-embed-text` | Embeddings / RAG | 0.3 GB | ★★★★★ |

Models are stored in `local_LLMs/models/ollama/` (portable).

---

## HuggingFace Engine (port 8001)

Supports any model from HuggingFace Hub. Uses 4-bit quantization (bitsandbytes nf4) by default.
Models cached to `local_LLMs/models/huggingface/`.

**Quantization options:** `4bit` (default) | `8bit` | `none`

**Model size guide for RTX 3050 6GB:**

| Model Size | Quantization | Fits in 6GB? |
|-----------|-------------|--------------|
| ≤ 7B params | 4-bit | ✓ fully GPU |
| 7–13B params | 4-bit | ✓ with minor CPU spill |
| 13–30B params | 4-bit | partial GPU + CPU offload (slow) |
| 30B+ params | any | ✗ CPU-only, very slow |

**VRAM coordination:** The router automatically unloads the opposite engine before each request. Only one model in VRAM at a time.

---

## Model Routing Tags

```
auto            → complexity-based auto routing
local:fast      → gemma3:4b
local:code      → qwen2.5-coder:7b
local:general   → mistral:7b
hf:<model_id>   → any HuggingFace model via HF engine
cloud:claude    → claude-opus-4-6  (needs ANTHROPIC_API_KEY)
cloud:openai    → gpt-4o           (needs OPENAI_API_KEY)
cloud:gemini    → gemini-2.0-flash (needs GOOGLE_API_KEY)
cloud:sonnet    → claude-sonnet-4-6
```

---

## Management Commands

```bash
# ── Start / Stop ─────────────────────────────────────────
bash start_router.sh --bg       # start Ollama + HF engine + router
bash start_router.sh --stop     # stop router + HF engine
bash start_hf_engine.sh --bg    # start HF engine only
bash start_hf_engine.sh --stop  # stop HF engine only
bash start_ollama.sh --background

# ── Ollama models ─────────────────────────────────────────
./ollama_bin/bin/ollama list
./ollama_bin/bin/ollama pull <model>
./ollama_bin/bin/ollama rm <model>

# ── HF engine ─────────────────────────────────────────────
curl http://localhost:8001/health
curl -X POST http://localhost:8001/v1/models/unload

# ── Migrate Ollama models to local folder (one-time) ──────
bash migrate_models.sh
```

---

## Recommended Models to Pull (RTX 3050 6GB)

```bash
OLLAMA=./ollama_bin/bin/ollama
$OLLAMA pull deepseek-r1:7b       # reasoning/analysis  4.9 GB
$OLLAMA pull phi3.5:3.8b          # fast triage         2.3 GB
$OLLAMA pull llama3.2:3b          # compact general     2.0 GB
$OLLAMA pull codellama:7b-instruct # code               3.8 GB
```
