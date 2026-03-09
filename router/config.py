"""
LLM Router — Configuration
Reads from .env file at project root or environment variables.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # ── Ollama ──────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_bin: str = "/home/ahm/Desktop/local_LLMs/ollama_bin/bin/ollama"

    # ── Cloud API keys (all optional — router falls back to local) ──────
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # ── Router server ────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Model routing table ──────────────────────────────────────────────
    # Override these to swap which model handles each tier
    model_trivial: str = "gemma3:4b"
    model_simple: str = "gemma3:4b"
    model_medium_code: str = "qwen2.5-coder:7b-instruct-q4_K_M"
    model_medium_general: str = "mistral:7b-instruct-q4_K_M"
    model_complex: str = "mistral:7b-instruct-q4_K_M"
    model_expert_cloud: str = "claude-opus-4-6"   # used when cloud key present
    model_expert_local: str = "mistral:7b-instruct-q4_K_M"  # fallback

    class Config:
        env_file = "/home/ahm/Desktop/local_LLMs/.env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
