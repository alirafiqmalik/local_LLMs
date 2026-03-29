"""
LLM Router — Configuration
Reads from .env file at project root or environment variables.
All paths are derived from BASE_DIR so the local_LLMs folder can be moved freely.
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional

# Resolve local_LLMs/ regardless of where this file is run from
BASE_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = str(BASE_DIR / ".env")


class Settings(BaseSettings):
    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    @property
    def ollama_bin(self) -> str:
        return str(BASE_DIR / "ollama_bin" / "bin" / "ollama")

    @property
    def ollama_models_dir(self) -> str:
        return str(BASE_DIR / "models" / "ollama")

    # ── HF Engine ───────────────────────────────────────────────────────────
    hf_engine_url:  str = "http://localhost:8001"
    hf_engine_port: int = 8001

    @property
    def hf_models_dir(self) -> str:
        return str(BASE_DIR / "models" / "huggingface")

    # ── HuggingFace token (for gated models) ────────────────────────────────
    hf_token: Optional[str] = None

    # ── Router server ────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Model routing table ──────────────────────────────────────────────────
    model_trivial:        str = "gemma3:4b"
    model_simple:         str = "gemma3:4b"
    model_medium_code:    str = "qwen2.5-coder:7b-instruct-q4_K_M"
    model_medium_general: str = "mistral:7b-instruct-q4_K_M"
    model_complex:        str = "mistral:7b-instruct-q4_K_M"
    model_expert:         str = "mistral:7b-instruct-q4_K_M"

    class Config:
        env_file          = _ENV_FILE
        env_file_encoding = "utf-8"
        extra             = "ignore"


settings = Settings()
