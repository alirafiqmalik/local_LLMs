"""
LLM Router — Complexity Classifier

Heuristic-first classification so we never waste a model call on routing.
Tiers:
  TRIVIAL  → gemma3:4b       (greetings, one-liners, simple lookups)
  SIMPLE   → gemma3:4b       (basic Q&A, short coding questions)
  MEDIUM   → qwen2.5/mistral (coding tasks, debugging, explanations)
  COMPLEX  → mistral:7b      (multi-step code, refactoring, analysis)
  EXPERT   → mistral:7b      (system design, long context — best local)
"""
import re
from enum import Enum
from typing import List, Dict


class Complexity(str, Enum):
    TRIVIAL = "trivial"
    SIMPLE  = "simple"
    MEDIUM  = "medium"
    COMPLEX = "complex"
    EXPERT  = "expert"


# ── Keyword banks ─────────────────────────────────────────────────────────────

TRIVIAL_REGEXES = [
    r"^(hi|hello|hey|sup|yo)[\s!.?]*$",
    r"^thanks?[\s!.?]*$",
    r"^(yes|no|ok|okay|sure|got it)[\s!.?]*$",
    r"^what is \w+[\s?]*$",
    r"^define \w+[\s?]*$",
    r"^who (is|was) \w+[\s?]*$",
]

EXPERT_KEYWORDS = [
    "system design", "microservices", "kubernetes", "distributed system",
    "high availability", "load balanc", "security audit", "penetration test",
    "production architecture", "scale to million", "entire codebase",
    "full stack application", "end-to-end", "devops pipeline", "ci/cd",
    "migrate entire", "overhaul", "comprehensive review",
]

COMPLEX_KEYWORDS = [
    "refactor", "design pattern", "algorithm", "performance optim",
    "concurren", "async", "multithreading", "race condition",
    "database schema", "api design", "test suite", "benchmark",
    "memory leak", "profil", "abstract", "architecture",
]

CODING_KEYWORDS = [
    "code", "function", "class", "method", "implement", "write a",
    "debug", "error", "bug", "fix", "syntax", "compile", "run",
    "script", "module", "import", "loop", "array", "dict", "list",
    "variable", "type", "interface", "struct", "enum", "async def",
    "python", "javascript", "typescript", "rust", "go ", "java ",
    "c++", "sql", "bash", "shell", "dockerfile", "yaml", "json",
]


def _token_count(text: str) -> int:
    return len(text.split())


def classify(messages: List[Dict]) -> tuple[Complexity, str]:
    """
    Returns (Complexity, reason_string).
    Pure heuristics — no LLM call needed.
    """
    user_parts = [
        m.get("content", "") for m in messages
        if m.get("role") in ("user", "system")
    ]
    text = " ".join(user_parts).strip()
    lower = text.lower()
    tokens = _token_count(text)

    # 1. Trivial: very short + matches pattern
    if tokens < 20:
        for pattern in TRIVIAL_REGEXES:
            if re.match(pattern, lower):
                return Complexity.TRIVIAL, f"trivial pattern match (tokens={tokens})"

    # 2. Expert: long context or architecture keywords
    expert_hits = [kw for kw in EXPERT_KEYWORDS if kw in lower]
    if expert_hits or tokens > 800:
        reason = f"expert keywords={expert_hits}" if expert_hits else f"very long prompt (tokens={tokens})"
        return Complexity.EXPERT, reason

    # 3. Complex: multiple complex signals
    complex_hits = [kw for kw in COMPLEX_KEYWORDS if kw in lower]
    if len(complex_hits) >= 2 or tokens > 350:
        return Complexity.COMPLEX, f"complex signals={complex_hits}, tokens={tokens}"

    # 4. Medium: coding task or moderate length
    code_hits = [kw for kw in CODING_KEYWORDS if kw in lower]
    if code_hits or len(complex_hits) == 1 or tokens > 60:
        task = "code" if len(code_hits) >= 2 else "general"
        return Complexity.MEDIUM, f"task_type={task}, code_hits={len(code_hits)}, tokens={tokens}"

    # 5. Simple default
    return Complexity.SIMPLE, f"short general query (tokens={tokens})"


def is_code_task(messages: List[Dict]) -> bool:
    text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    ).lower()
    return sum(1 for kw in CODING_KEYWORDS if kw in text) >= 2
