"""
HF Engine — Model Loader

Handles two backends:
  - transformers + bitsandbytes  → any HuggingFace model ID
  - llama-cpp-python             → local .gguf files

Only one model is held in memory at a time.
Call unload() before loading a new model, or load() will auto-swap.
"""
import gc
import os
import time
import threading
from pathlib import Path
from typing import Optional

import torch

# BASE_DIR = local_LLMs/ — works wherever the folder is moved
BASE_DIR      = Path(__file__).resolve().parent.parent.parent
HF_MODELS_DIR = BASE_DIR / "models" / "huggingface"
HF_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Internal state ────────────────────────────────────────────────────────────

class _State:
    model_id:   str   = None
    backend:    str   = None   # "transformers" | "llama_cpp"
    model:      object = None
    tokenizer:  object = None  # None for llama_cpp
    loaded_at:  float  = None

_current: Optional[_State] = None
_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_gguf(model_id: str) -> bool:
    low = model_id.lower()
    return low.endswith(".gguf") or "gguf" in low.split("/")[-1]


def vram_free_gb() -> float:
    try:
        free, _ = torch.cuda.mem_get_info()
        return free / 1024 ** 3
    except Exception:
        return 6.0


def vram_used_gb() -> float:
    try:
        return torch.cuda.memory_allocated() / 1024 ** 3
    except Exception:
        return 0.0


def _pick_quantization(model_id: str, requested: str) -> str:
    """Auto-select quantization if caller passes 'auto'."""
    if requested != "auto":
        return requested
    free = vram_free_gb()
    if free >= 7:
        return "8bit"
    return "4bit"


# ── Load ──────────────────────────────────────────────────────────────────────

def load(model_id: str, quantization: str = "4bit") -> dict:
    """
    Load a model.  Thread-safe — blocks until loaded.

    model_id:
      - HuggingFace repo: "microsoft/Phi-4", "Qwen/Qwen2.5-Coder-7B-Instruct"
      - Local .gguf path: "/abs/path/model.gguf" or relative to models/huggingface/
    quantization:
      - "4bit"  → bitsandbytes nf4  (recommended, fits 7B in ~4GB)
      - "8bit"  → bitsandbytes int8
      - "none"  → fp16 (needs full VRAM, use only for tiny models)
      - "auto"  → pick based on free VRAM
    """
    global _current
    with _lock:
        if _current and _current.model_id == model_id:
            return {"status": "already_loaded", "model_id": model_id, "backend": _current.backend}

        if _current:
            _do_unload()

        quantization = _pick_quantization(model_id, quantization)

        if is_gguf(model_id):
            _load_gguf(model_id)
        else:
            _load_transformers(model_id, quantization)

        return {
            "status": "loaded",
            "model_id": _current.model_id,
            "backend": _current.backend,
            "quantization": quantization,
            "vram_used_gb": round(vram_used_gb(), 2),
        }


def _load_transformers(model_id: str, quantization: str):
    global _current
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token      = os.environ.get("HF_TOKEN")
    cache_dir  = str(HF_MODELS_DIR)

    # Quantization config
    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,   # nested quant — saves extra ~10% VRAM
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        token=token,
        quantization_config=bnb_config,
        device_map="auto",            # spills overflow layers to CPU automatically
        torch_dtype=torch.float16 if quantization == "none" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    s = _State()
    s.model_id  = model_id
    s.backend   = "transformers"
    s.model     = model
    s.tokenizer = tokenizer
    s.loaded_at = time.time()
    _current = s


def _load_gguf(model_id: str):
    global _current
    from llama_cpp import Llama

    # Resolve path — absolute or relative to HF_MODELS_DIR
    path = Path(model_id)
    if not path.is_absolute():
        path = HF_MODELS_DIR / model_id
    if not path.exists():
        raise FileNotFoundError(f"GGUF not found: {path}")

    file_size_gb = path.stat().st_size / 1024 ** 3
    free = vram_free_gb()
    # Estimate GPU layers: if model fits, offload all; else scale proportionally
    n_gpu_layers = -1 if file_size_gb <= free * 0.9 else max(0, int((free / file_size_gb) * 32))

    llm = Llama(
        model_path=str(path),
        n_ctx=4096,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )

    s = _State()
    s.model_id  = str(path)
    s.backend   = "llama_cpp"
    s.model     = llm
    s.tokenizer = None
    s.loaded_at = time.time()
    _current = s


# ── Unload ────────────────────────────────────────────────────────────────────

def unload() -> dict:
    global _current
    with _lock:
        if not _current:
            return {"status": "no_model_loaded"}
        mid = _current.model_id
        _do_unload()
        return {"status": "unloaded", "model_id": mid}


def _do_unload():
    """Must be called with _lock held."""
    global _current
    if _current is None:
        return
    del _current.model
    if _current.tokenizer is not None:
        del _current.tokenizer
    _current = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ── Status ────────────────────────────────────────────────────────────────────

def status() -> dict:
    if _current is None:
        return {"loaded": False, "model_id": None, "backend": None}
    vram = {}
    if torch.cuda.is_available():
        vram = {
            "used_gb":  round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
            "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2),
        }
    return {
        "loaded":     True,
        "model_id":   _current.model_id,
        "backend":    _current.backend,
        "loaded_at":  _current.loaded_at,
        "vram":       vram,
    }


# ── Generate ──────────────────────────────────────────────────────────────────

def generate(
    messages:     list,
    max_tokens:   int   = 2048,
    temperature:  float = 0.7,
    stream:       bool  = False,
):
    if _current is None:
        raise RuntimeError("No model loaded. Call load() first.")

    if _current.backend == "transformers":
        return _gen_transformers(messages, max_tokens, temperature, stream)
    return _gen_llama_cpp(messages, max_tokens, temperature, stream)


def _gen_transformers(messages, max_tokens, temperature, stream):
    from transformers import TextIteratorStreamer

    model     = _current.model
    tokenizer = _current.tokenizer

    # Build prompt via chat template (works for almost all modern models)
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Generic fallback
        prompt = "\n".join(
            f"<|{m['role']}|>\n{m['content']}" for m in messages
        ) + "\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

    gen_kwargs = {
        "max_new_tokens":  max_tokens,
        "do_sample":       temperature > 0,
        "pad_token_id":    tokenizer.eos_token_id,
        "repetition_penalty": 1.1,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    if stream:
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer

        def _run():
            with torch.no_grad():
                model.generate(**inputs, **gen_kwargs)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return streamer  # caller iterates

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def _gen_llama_cpp(messages, max_tokens, temperature, stream):
    llm = _current.model
    result = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )
    if stream:
        return result  # generator of chunks
    return result["choices"][0]["message"]["content"]
