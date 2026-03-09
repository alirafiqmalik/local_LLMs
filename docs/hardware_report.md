# Hardware Capabilities & Local LLM Strategy

## System Summary

| Component | Spec | LLM Role |
|-----------|------|----------|
| **NVIDIA RTX 3050** | 6GB VRAM, CUDA 13.1 | **Primary inference GPU** — runs all quantized models |
| **Intel Iris Xe** | Shared RAM (iGPU), 13th Gen | **Deferred** — SYCL/IPEX-LLM possible but slower than NVIDIA |
| **Intel i5-13500H** | 12 cores / 16 threads, 4.7GHz | CPU offload fallback, concurrent light inference |
| **RAM** | 16GB + 16GB swap | Supports CPU-offloaded layers when VRAM is full |
| **Storage** | 683GB free NVMe | Ample for dozens of model files |

## VRAM Constraint (6GB) — What Fits

| Category | Max Model Size | Examples |
|----------|---------------|----------|
| **Fully GPU-resident** | 7–8B params @ Q4 quantization | Qwen2.5-Coder 7B, Mistral 7B |
| **Comfortable fit** | 3–4B params @ FP16/Q8 | Gemma3 4B, Phi-3 Mini |
| **Requires CPU offload** | 13B+ params | Slow (~2–5 tok/s vs ~50+ on GPU) |

## How We Used Them

```
┌──────────────────────────────────────────┐
│          Ollama v0.16.3 Server           │
│     http://localhost:11434/v1            │
│     (OpenAI-compatible API)              │
├──────────────┬───────────────────────────┤
│  RTX 3050    │  Models (time-shared)     │
│  6GB CUDA    │  • qwen2.5-coder 7B (4.7GB) │
│              │  • mistral 7B      (4.4GB) │
│              │  • gemma3 4B       (3.3GB) │
│              │  • nomic-embed     (274MB) │
├──────────────┼───────────────────────────┤
│  i5-13500H   │  Fallback for concurrent  │
│  16 threads  │  inference if GPU is busy  │
├──────────────┼───────────────────────────┤
│  Iris Xe     │  Reserved (future SYCL    │
│  iGPU        │  backend for 2nd model)   │
└──────────────┴───────────────────────────┘
```

## Key Design Decisions

1. **Ollama over raw llama.cpp** — simpler management, built-in OpenAI API, auto CUDA detection
2. **Q4_K_M quantization** — best quality-to-VRAM ratio; fits 7B models in ~5GB
3. **User-space install** — no sudo required, portable to `~/Desktop/local_LLMs/`
4. **Intel iGPU deferred** — NVIDIA handles everything; iGPU adds complexity for marginal gain
5. **Model time-sharing** — Ollama swaps models in/out of GPU; only 1 loaded at a time by default
