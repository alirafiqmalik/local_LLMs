"""Shared client and helpers for example scripts."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

ROUTER_V1 = "http://localhost:8000/v1"
ROUTER_API = "http://localhost:8000"
HF_ENGINE = "http://localhost:8001"

client = OpenAI(base_url=ROUTER_V1, api_key="local")

# Preference order: first name found in Ollama `local_installed` wins.
_FAST_PREFS = (
    "gemma3:4b",
    "qwen3:4b",
    "phi4-mini:latest",
)
_CODE_PREFS = (
    "qwen2.5-coder:7b-instruct-q4_K_M",
    "qwen2.5-coder:7b",
)
_GENERAL_PREFS = (
    "mistral:7b",
    "mistral:7b-instruct-q4_K_M",
    "llama3.1:8b",
    "deepseek-r1:7b",
)
_EMBED_PREFS = ("nomic-embed-text",)


@dataclass(frozen=True)
class ExampleModels:
    """Concrete Ollama model names taken from the router’s installed list."""

    fast: str
    code: str
    general: str
    embed: str | None
    installed: frozenset[str]
    routing_table: dict[str, str]


def _pick(installed: set[str], prefs: tuple[str, ...], role: str) -> str:
    for name in prefs:
        if name in installed:
            return name
    print(
        f"None of {prefs} are installed (need a {role} model). "
        f"Installed: {sorted(installed)}",
        file=sys.stderr,
    )
    sys.exit(1)


def get_example_models() -> ExampleModels:
    """Resolve models from GET /api/models so examples never call missing Ollama tags."""
    try:
        data = httpx.get(f"{ROUTER_API}/api/models", timeout=8).json()
    except Exception as e:
        print(
            f"Cannot reach router at {ROUTER_API}/api/models ({e}). "
            "Start with: ./llm start",
            file=sys.stderr,
        )
        sys.exit(1)
    rows = data.get("local_installed") or []
    installed = {m["id"] for m in rows if m.get("id")}
    if not installed:
        print("No Ollama models reported under local_installed.", file=sys.stderr)
        sys.exit(1)
    rt = data.get("routing_table") or {}
    embed = None
    for name in _EMBED_PREFS:
        if name in installed:
            embed = name
            break
    return ExampleModels(
        fast=_pick(installed, _FAST_PREFS, "fast"),
        code=_pick(installed, _CODE_PREFS, "code"),
        general=_pick(installed, _GENERAL_PREFS, "general"),
        embed=embed,
        installed=frozenset(installed),
        routing_table=dict(rt),
    )


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print("─" * 60)


def _truncate(s: Any, max_len: int = 72) -> str:
    if s is None:
        return ""
    t = str(s).replace("\n", " ")
    return t if len(t) <= max_len else t[: max_len - 3] + "..."


def router_meta(resp) -> None:
    extra = getattr(resp, "model_extra", None) or {}
    r = extra.get("_router", {})
    print(f"  req_id     : {r.get('req_id')}")
    print(f"  complexity : {r.get('complexity')}")
    print(f"  classify   : {_truncate(r.get('classify_reason'))}")
    print(f"  routing    : {_truncate(r.get('routing_reason'))}")
    print(f"  model used : {r.get('model_used')}")
    print(f"  provider   : {r.get('provider')}")
    print(f"  latency    : {r.get('latency_ms')}ms")


def pretty_json(data: dict, indent: int = 6) -> str:
    return json.dumps(data, indent=indent)
