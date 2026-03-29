# LLM Router — Model Guide

How to download, manage, and use models with the local LLM stack.

---

## Ollama Models

### Download a model

```bash
./llm pull <tag>                          # use shorthand (see below)
./llm pull fast                           # phi3.5:3.8b
./llm pull code                           # qwen2.5-coder:7b-instruct-q4_K_M
./llm pull general                        # mistral:7b-instruct-q4_K_M
./llm pull reasoning                      # deepseek-r1:7b
./llm pull gemma3:4b                      # any ollama model tag directly
```

Models are stored in `models/ollama/` (set automatically via `OLLAMA_MODELS`).

### Recommended models for RTX 3050 6GB

| Model | Size | Role | Pull tag |
|-------|------|------|----------|
| `gemma3:4b` | 3.3 GB | Fast triage (installed) | `./llm pull gemma3:4b` |
| `qwen2.5-coder:7b-instruct-q4_K_M` | 4.7 GB | Code gen (installed) | `./llm pull code` |
| `mistral:7b-instruct-q4_K_M` | 4.4 GB | Reasoning (installed) | `./llm pull general` |
| `nomic-embed-text` | 0.3 GB | Embeddings (installed) | `./llm pull nomic-embed-text` |
| `phi3.5:3.8b` | 2.3 GB | Faster triage upgrade | `./llm pull fast` |
| `deepseek-r1:7b` | 4.9 GB | Reasoning / analysis | `./llm pull reasoning` |
| `llama3.2:3b` | 2.0 GB | Compact general | `./llm pull llama3.2:3b` |
| `codellama:7b-instruct` | 3.8 GB | Code completion | `./llm pull codellama:7b-instruct` |

**Note:** Never load two 7B models simultaneously — total VRAM is 6GB.

### List / remove models

```bash
./llm models                                # list installed models
./ollama_bin/bin/ollama rm <model>          # delete a model
du -sh models/ollama/                       # total disk usage
```

### Override which model handles each complexity tier

Edit `.env` and uncomment the overrides:

```bash
MODEL_TRIVIAL=phi3.5:3.8b           # faster triage
MODEL_MEDIUM_CODE=codellama:7b-instruct
MODEL_COMPLEX=deepseek-r1:7b        # better reasoning
```

Then restart: `./llm restart`

---

## HuggingFace Models

HF models run on a separate engine at port 8001. Supported backends:
- **transformers + bitsandbytes** — any HF Hub model (auto 4-bit quantization)
- **llama-cpp-python** — local `.gguf` files

Models are cached to `models/huggingface/`.

### Use any HF Hub model (via router)

Just prefix the model ID with `hf:` — it downloads and loads on first call:

```python
client.chat.completions.create(
    model="hf:Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "hello"}]
)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hf:mistralai/Mistral-7B-Instruct-v0.3","messages":[{"role":"user","content":"hello"}]}'
```

**First call downloads + loads the model** (30–120s depending on model size).

### Pre-download a model

```bash
HF_HOME=models/huggingface huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

### Quantization options

Control quantization via the HF engine API directly:

```bash
curl -X POST http://localhost:8001/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen2.5-7B-Instruct", "quantization": "4bit"}'
```

| Option | VRAM usage | Use when |
|--------|-----------|---------|
| `4bit` (default) | ~4.5 GB for 7B | fits in 6GB |
| `8bit` | ~7 GB for 7B | too large for 4bit |
| `none` | ~14 GB for 7B | CPU-only, very slow |

### Model size guide for RTX 3050 6GB

| Model size | 4-bit fits in 6GB? |
|-----------|--------------------|
| ≤ 7B | Yes, fully GPU |
| 7–13B | Yes, minor CPU spill |
| 13–30B | Partial GPU + CPU (slow) |
| 30B+ | CPU-only, very slow |

### Gated models (Llama-3, Phi-4, Gemma-3)

Add your HuggingFace token to `.env`:

```bash
HF_TOKEN=hf_xxxx
```

### GGUF files (local, no download)

Place a `.gguf` file in `models/huggingface/` and reference it by filename:

```python
client.chat.completions.create(model="hf:my-model.gguf", ...)
```

### Recommended HF models for RTX 3050 6GB

| Model | Size (4-bit) | Role |
|-------|-------------|------|
| `Qwen/Qwen2.5-7B-Instruct` | ~4.5 GB | Strong general |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | ~4.5 GB | Code |
| `mistralai/Mistral-7B-Instruct-v0.3` | ~4.3 GB | Instruction |
| `microsoft/Phi-3.5-mini-instruct` | ~2.5 GB | Fast + capable |
| `Orion-zhen/Qwen2.5-7B-Instruct-Uncensored` | ~4.5 GB | Uncensored |
| `google/gemma-2-9b-it` | ~5.5 GB | Google's 9B |

### Unload HF model (free VRAM)

```bash
curl -X POST http://localhost:8001/v1/models/unload
```

---

## Storage locations

| Engine | Path |
|--------|------|
| Ollama models | `models/ollama/` |
| HuggingFace cache | `models/huggingface/` |
| GGUF files | `models/huggingface/*.gguf` |

```bash
du -sh models/ollama/ models/huggingface/
```
