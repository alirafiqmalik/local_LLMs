"""
LLM Router — Model Registry & Routing Logic

Routing priority:
  1. Explicit model name in request → use that directly
  2. Virtual routing tags: "auto", "local:fast", "local:code", "cloud:claude", etc.
  3. Complexity-based auto-routing
"""
from dataclasses import dataclass
from typing import List, Dict
from classifier import Complexity, is_code_task
from config import settings


@dataclass
class ModelInfo:
    name: str
    provider: str          # "ollama" | "cloud_anthropic" | "cloud_openai" | "cloud_google"
    vram_gb: float
    tier: str              # "fast" | "code" | "general" | "reasoning" | "embed" | "cloud"
    description: str
    recommended_for: List[str]


# ── Model registry ────────────────────────────────────────────────────────────

LOCAL_MODELS: Dict[str, ModelInfo] = {
    "gemma3:4b": ModelInfo(
        name="gemma3:4b",
        provider="ollama",
        vram_gb=3.3,
        tier="fast",
        description="Google Gemma 3 4B — fast, lightweight, great triage model",
        recommended_for=["trivial", "simple", "quick chat"],
    ),
    "qwen2.5-coder:7b-instruct-q4_K_M": ModelInfo(
        name="qwen2.5-coder:7b-instruct-q4_K_M",
        provider="ollama",
        vram_gb=4.7,
        tier="code",
        description="Qwen 2.5 Coder 7B — best local code gen, debugging, refactoring",
        recommended_for=["coding", "debugging", "code review"],
    ),
    "mistral:7b-instruct-q4_K_M": ModelInfo(
        name="mistral:7b-instruct-q4_K_M",
        provider="ollama",
        vram_gb=4.4,
        tier="general",
        description="Mistral 7B — strong reasoning, planning, analysis",
        recommended_for=["analysis", "planning", "complex reasoning"],
    ),
    "nomic-embed-text": ModelInfo(
        name="nomic-embed-text",
        provider="ollama",
        vram_gb=0.3,
        tier="embed",
        description="Nomic Embed — fast text embeddings for RAG/search",
        recommended_for=["embeddings", "RAG", "semantic search"],
    ),
}

# Models recommended to pull for coding/analysis focus on RTX 3050 6GB
RECOMMENDED_MODELS: Dict[str, ModelInfo] = {
    "deepseek-r1:7b": ModelInfo(
        name="deepseek-r1:7b",
        provider="ollama",
        vram_gb=4.9,
        tier="reasoning",
        description="DeepSeek R1 7B — chain-of-thought reasoning, analysis (~4.9GB Q4)",
        recommended_for=["analysis", "math", "reasoning", "debugging"],
    ),
    "phi4:latest": ModelInfo(
        name="phi4:latest",
        provider="ollama",
        vram_gb=8.9,
        tier="general",
        description="Microsoft Phi-4 14B — very capable but requires CPU offload on 6GB",
        recommended_for=["general reasoning"],
    ),
    "phi3.5:3.8b": ModelInfo(
        name="phi3.5:3.8b",
        provider="ollama",
        vram_gb=2.3,
        tier="fast",
        description="Microsoft Phi-3.5 Mini 3.8B — ultra-fast, 2.3GB, great for triage",
        recommended_for=["trivial", "simple", "fast"],
    ),
    "llama3.2:3b": ModelInfo(
        name="llama3.2:3b",
        provider="ollama",
        vram_gb=2.0,
        tier="fast",
        description="Meta Llama 3.2 3B — compact, fast, good general knowledge",
        recommended_for=["trivial", "simple", "chat"],
    ),
    "starcoder2:7b": ModelInfo(
        name="starcoder2:7b",
        provider="ollama",
        vram_gb=4.5,
        tier="code",
        description="StarCoder2 7B — dedicated code completion, multi-language",
        recommended_for=["code completion", "fill-in-middle"],
    ),
    "codellama:7b-instruct": ModelInfo(
        name="codellama:7b-instruct",
        provider="ollama",
        vram_gb=3.8,
        tier="code",
        description="Meta Code Llama 7B — solid code model, instruction-tuned",
        recommended_for=["coding", "code explanation"],
    ),
    "qwen2.5:7b": ModelInfo(
        name="qwen2.5:7b",
        provider="ollama",
        vram_gb=4.7,
        tier="general",
        description="Qwen 2.5 7B general — strong on analysis and multilingual",
        recommended_for=["analysis", "general", "multilingual"],
    ),
}

CLOUD_MODELS: Dict[str, ModelInfo] = {
    "claude-opus-4-6": ModelInfo(
        name="claude-opus-4-6",
        provider="cloud_anthropic",
        vram_gb=0,
        tier="cloud",
        description="Claude Opus 4.6 — best reasoning, complex code, architecture",
        recommended_for=["expert", "system design", "complex analysis"],
    ),
    "claude-sonnet-4-6": ModelInfo(
        name="claude-sonnet-4-6",
        provider="cloud_anthropic",
        vram_gb=0,
        tier="cloud",
        description="Claude Sonnet 4.6 — balanced speed/quality for complex tasks",
        recommended_for=["complex", "code review", "analysis"],
    ),
    "gpt-4o": ModelInfo(
        name="gpt-4o",
        provider="cloud_openai",
        vram_gb=0,
        tier="cloud",
        description="OpenAI GPT-4o — strong multimodal, coding, reasoning",
        recommended_for=["expert", "coding", "vision"],
    ),
    "gemini-2.0-flash": ModelInfo(
        name="gemini-2.0-flash",
        provider="cloud_google",
        vram_gb=0,
        tier="cloud",
        description="Gemini 2.0 Flash — fast, large context, good at analysis",
        recommended_for=["analysis", "long context", "complex"],
    ),
}

ALL_MODELS = {**LOCAL_MODELS, **CLOUD_MODELS}

# ── Virtual routing tags → (provider, model) ─────────────────────────────────
VIRTUAL_ROUTES = {
    "local:fast":    ("ollama", "gemma3:4b"),
    "local:code":    ("ollama", "qwen2.5-coder:7b-instruct-q4_K_M"),
    "local:general": ("ollama", "mistral:7b-instruct-q4_K_M"),
    "cloud:claude":  ("cloud_anthropic", "claude-opus-4-6"),
    "cloud:openai":  ("cloud_openai", "gpt-4o"),
    "cloud:gemini":  ("cloud_google", "gemini-2.0-flash"),
    "cloud:sonnet":  ("cloud_anthropic", "claude-sonnet-4-6"),
}


def _cloud_available() -> tuple[bool, bool, bool]:
    """Returns (anthropic, openai, google) availability."""
    return (
        bool(settings.anthropic_api_key),
        bool(settings.openai_api_key),
        bool(settings.google_api_key),
    )


def resolve_model(
    requested_model: str,
    complexity: Complexity,
    messages: List[Dict],
    task_type: str = "general",
) -> tuple[str, str, str]:
    """
    Returns (provider, model_name, routing_reason).

    Virtual model tags:
      "auto"          — full auto routing
      "local:fast"    — force fastest local model
      "local:code"    — force code model
      "local:general" — force general local model
      "cloud:claude"  — force Claude (requires ANTHROPIC_API_KEY)
      "cloud:openai"  — force GPT-4o (requires OPENAI_API_KEY)
      "cloud:gemini"  — force Gemini (requires GOOGLE_API_KEY)
    """
    model = (requested_model or "auto").strip()

    # ── 1. Virtual routing tags ───────────────────────────────────────────
    if model in VIRTUAL_ROUTES:
        provider, name = VIRTUAL_ROUTES[model]
        return provider, name, f"explicit tag: {model}"

    # ── 2. Direct model name (known local or cloud) ───────────────────────
    if model in LOCAL_MODELS:
        return "ollama", model, f"direct: {model}"
    if model in CLOUD_MODELS:
        info = CLOUD_MODELS[model]
        return info.provider, model, f"direct: {model}"

    # ── 3. Auto-routing by complexity ─────────────────────────────────────
    has_anthropic, has_openai, has_google = _cloud_available()
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

    # EXPERT — prefer cloud, fallback to best local
    if has_anthropic:
        return "cloud_anthropic", settings.model_expert_cloud, f"{reason}, cloud=claude"
    if has_openai:
        return "cloud_openai", "gpt-4o", f"{reason}, cloud=openai"
    if has_google:
        return "cloud_google", "gemini-2.0-flash", f"{reason}, cloud=gemini"

    # No cloud keys — use best local
    return "ollama", settings.model_expert_local, f"{reason}, no cloud keys → best local"
