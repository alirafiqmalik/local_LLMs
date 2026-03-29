"""
LLM Router — Model Registry & Routing Logic

Routing priority:
  1. HF prefix "hf:<model_id>" → route to HF engine
  2. Virtual routing tags: "local:fast", "local:code", "local:general"
  3. Any other explicit name (e.g. mistral:7b, llama3.1:8b) → Ollama as-is
  4. "auto" only → complexity-based routing (LOCAL_MODELS is metadata for /api/models, not a gate)
"""
from dataclasses import dataclass
from typing import List, Dict
from classifier import Complexity, is_code_task
from config import settings


@dataclass
class ModelInfo:
    name: str
    provider: str   # "ollama" | "hf_engine"
    vram_gb: float
    description: str
    recommended_for: List[str]


# ── Model registry ────────────────────────────────────────────────────────────

LOCAL_MODELS: Dict[str, ModelInfo] = {
    "gemma3:4b": ModelInfo(
        name="gemma3:4b",
        provider="ollama",
        vram_gb=3.3,
        description="Google Gemma 3 4B — fast, lightweight, great triage model",
        recommended_for=["trivial", "simple", "quick chat"],
    ),
    "qwen2.5-coder:7b-instruct-q4_K_M": ModelInfo(
        name="qwen2.5-coder:7b-instruct-q4_K_M",
        provider="ollama",
        vram_gb=4.7,
        description="Qwen 2.5 Coder 7B — best local code gen, debugging, refactoring",
        recommended_for=["coding", "debugging", "code review"],
    ),
    "mistral:7b-instruct-q4_K_M": ModelInfo(
        name="mistral:7b-instruct-q4_K_M",
        provider="ollama",
        vram_gb=4.4,
        description="Mistral 7B — strong reasoning, planning, analysis",
        recommended_for=["analysis", "planning", "complex reasoning"],
    ),
    "nomic-embed-text": ModelInfo(
        name="nomic-embed-text",
        provider="ollama",
        vram_gb=0.3,
        description="Nomic Embed — fast text embeddings for RAG/search",
        recommended_for=["embeddings", "RAG", "semantic search"],
    ),
}

ALL_MODELS = LOCAL_MODELS

# ── Virtual routing tags → (provider, model) ─────────────────────────────────
VIRTUAL_ROUTES = {
    "local:fast":    ("ollama", "gemma3:4b"),
    "local:code":    ("ollama", "qwen2.5-coder:7b-instruct-q4_K_M"),
    "local:general": ("ollama", "mistral:7b-instruct-q4_K_M"),
}

# ── HF prefix ─────────────────────────────────────────────────────────────────
HF_PREFIX = "hf:"   # e.g. "hf:microsoft/Phi-4" or "hf:Qwen/Qwen2.5-7B-Instruct"


def resolve_model(
    requested_model: str,
    complexity: Complexity,
    messages: List[Dict],
    task_type: str = "general",
) -> tuple[str, str, str]:
    """
    Returns (provider, model_name, routing_reason).

    Virtual model tags:
      "auto"                    — full auto routing (always local)
      "local:fast"              — force fastest local Ollama model
      "local:code"              — force Ollama code model
      "local:general"           — force Ollama general model
      "hf:<model_id>"           — force HF engine  e.g. "hf:microsoft/Phi-4"
      "<ollama tag>"            — any installed Ollama model id, passed through
    """
    model = (requested_model or "auto").strip()

    # ── 1. HF prefix → always HF engine ──────────────────────────────────
    if model.startswith(HF_PREFIX):
        hf_model_id = model[len(HF_PREFIX):]
        return "hf_engine", hf_model_id, f"explicit hf: prefix → {hf_model_id}"

    # ── 2. Virtual routing tags ───────────────────────────────────────────
    if model in VIRTUAL_ROUTES:
        provider, name = VIRTUAL_ROUTES[model]
        return provider, name, f"explicit tag: {model}"

    # ── 3. Explicit Ollama model id (any tag installed in Ollama)
    if model != "auto":
        return "ollama", model, f"direct: {model}"

    # ── 4. Auto-routing by complexity ("auto" only) ───────────────────────
    reason = f"auto, complexity={complexity.value}"

    if complexity == Complexity.TRIVIAL:
        return "ollama", settings.model_trivial, reason

    if complexity == Complexity.SIMPLE:
        return "ollama", settings.model_simple, reason

    if complexity == Complexity.MEDIUM:
        coding = task_type == "code" or is_code_task(messages)
        m = settings.model_medium_code if coding else settings.model_medium_general
        return "ollama", m, f"{reason}, {'code' if coding else 'general'}"

    if complexity == Complexity.COMPLEX:
        return "ollama", settings.model_complex, reason

    # EXPERT — best local model
    return "ollama", settings.model_expert, f"{reason}, best local"
