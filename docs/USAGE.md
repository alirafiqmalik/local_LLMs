# LLM Router — Usage

**Base URL:** `http://localhost:8000/v1`
**Auth:** any string (e.g. `"local"`)
**Protocol:** OpenAI-compatible

---

## Start

```bash
./llm start           # Start Ollama + router
./llm start --hf      # Also start HF engine (port 8001)
# Dashboard: http://localhost:8000/
```

Activate the router venv (from project root):

```bash
source .venv/bin/activate
```

---

## Call via OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

resp = client.chat.completions.create(
    model="auto",   # router picks the right model
    messages=[{"role": "user", "content": "your prompt here"}]
)
print(resp.choices[0].message.content)

# Routing decision is in the response
print(resp.model_extra["_router"])
# → {"complexity": "medium", "model_used": "qwen2.5-coder:7b...", "latency_ms": 1200}
```

---

## Model Routing Tags

| `model=` value | Routes to |
|----------------|-----------|
| `"auto"` | auto-detected by complexity |
| `"local:fast"` | gemma3:4b (fastest) |
| `"local:code"` | qwen2.5-coder:7b |
| `"local:general"` | mistral:7b |
| `"hf:Qwen/Qwen2.5-7B-Instruct"` | any HuggingFace model |
| `"hf:model.gguf"` | local GGUF file in `models/huggingface/` |
| `"cloud:claude"` | Claude Opus 4.6 |
| `"cloud:openai"` | GPT-4o |
| `"cloud:gemini"` | Gemini 2.0 Flash |
| `"gemma3:4b"` | direct model name (bypasses routing) |

---

## curl

```bash
# Auto-routed
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}]}'

# Force HuggingFace model
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hf:microsoft/Phi-4","messages":[{"role":"user","content":"explain recursion"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local:code","stream":true,"messages":[{"role":"user","content":"fizzbuzz in python"}]}'
```

---

## Hints

```python
# Bias toward code model for ambiguous prompts
client.chat.completions.create(
    model="auto",
    messages=[...],
    extra_body={"task_type": "code"}   # "code" | "analysis" | "general"
)
```

---

## Embeddings

```python
emb = client.embeddings.create(model="nomic-embed-text", input="search query")
vector = emb.data[0].embedding  # 768-dim
```

---

## Status & Health

```bash
./llm status                           # full health check (JSON)
curl http://localhost:8000/api/status  # router + ollama + hf engine health
curl http://localhost:8000/api/models  # local_installed (all Ollama /api/tags) + routing_table
curl http://localhost:8001/health      # hf engine direct
```

`GET /api/models` returns JSON shaped like:

- **`local_installed`** — every model from Ollama `GET /api/tags` (same host as `OLLAMA_URL` / `ollama_base_url`). Each object has `id` (Ollama model name), plus `size`, `modified_at`, `digest` when present. If the name fuzzy-matches `LOCAL_MODELS`, the row also includes `vram_gb`, `description`, and `recommended_for`.
- **`routing_table`** — complexity → model id mapping from settings.

---

## Configure via `.env`

```bash
HF_TOKEN=hf_xxxx             # for gated HF models (Llama-3, Phi-4, Gemma-3)
ANTHROPIC_API_KEY=sk-ant-... # enables cloud:claude for expert routing
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```
