# Local LLM Configuration Guide

## API Endpoints

| Endpoint | URL | Notes |
|----------|-----|-------|
| **Ollama Native** | `http://localhost:11434/api` | Full Ollama API |
| **OpenAI Compatible** | `http://localhost:11434/v1` | Drop-in replacement for OpenAI SDK |

No API key required. Set `OPENAI_API_BASE=http://localhost:11434/v1` in your tools.

---

## Installed Models & Recommended Roles

| Model | Best For | VRAM | Speed |
|-------|----------|------|-------|
| `qwen2.5-coder:7b-instruct-q4_K_M` | Code generation, debugging, refactoring | ~5GB | ★★★ |
| `mistral:7b-instruct-q4_K_M` | Planning, reasoning, summarization | ~5GB | ★★★ |
| `gemma3:4b` | Fast tasks, triage, lightweight chat | ~3GB | ★★★★★ |
| `nomic-embed-text` | RAG embeddings, semantic search | ~0.3GB | ★★★★★ |

---

## Multi-Agent Role Assignment

```
┌─────────────────────────────────────────────────────┐
│                  ORCHESTRATOR                        │
│          (mistral:7b-instruct-q4_K_M)               │
│       Plans tasks, coordinates agents                │
└─────────────┬───────────────────┬───────────────────┘
              │                   │
    ┌─────────▼──────┐  ┌────────▼─────────┐
    │  CODE AGENT     │  │  REVIEW AGENT    │
    │  qwen2.5-coder  │  │  gemma3:4b       │
    │  Writes code    │  │  Quick reviews   │
    └────────────────┘  └──────────────────┘
```

### OpenClaw / OpenAI SDK Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # any string works
)

# Code generation
response = client.chat.completions.create(
    model="qwen2.5-coder:7b-instruct-q4_K_M",
    messages=[{"role": "user", "content": "Write a sorting function"}]
)

# Embeddings
embedding = client.embeddings.create(
    model="nomic-embed-text",
    input="Search query here"
)
```

### curl Examples

```bash
# Chat completion
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma3:4b", "messages": [{"role": "user", "content": "Hello"}]}'

# Embeddings
curl http://localhost:11434/api/embed \
  -d '{"model": "nomic-embed-text", "input": "search query"}'
```

---

## Concurrent Inference Notes

- Ollama auto-swaps models in GPU memory (only 1 model loaded at a time by default)
- To keep multiple models loaded: `OLLAMA_MAX_LOADED_MODELS=2` (only if models fit combined)
- `gemma3:4b` (3GB) + `nomic-embed-text` (0.3GB) = 3.3GB → fits in 6GB VRAM together
- Larger models must time-share the GPU

---

## Management Commands

```bash
# Start server
bash ~/Desktop/local_LLMs/start_ollama.sh --background

# List models
~/Desktop/local_LLMs/ollama_bin/bin/ollama list

# Run interactive chat
~/Desktop/local_LLMs/ollama_bin/bin/ollama run gemma3:4b

# Pull new model
~/Desktop/local_LLMs/ollama_bin/bin/ollama pull <model>

# Remove model
~/Desktop/local_LLMs/ollama_bin/bin/ollama rm <model>

# Test all models
bash ~/Desktop/local_LLMs/test_models.sh
```
