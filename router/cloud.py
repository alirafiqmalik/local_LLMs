"""
LLM Router — Cloud API Clients

Wrappers for Anthropic (Claude), OpenAI, and Google (Gemini).
All return OpenAI-compatible response dicts or async generators for streaming.
"""
import json
import httpx
from typing import AsyncGenerator, List, Dict, Any, Union
from config import settings


# ── Anthropic / Claude ────────────────────────────────────────────────────────

async def call_anthropic(
    messages: List[Dict],
    model: str,
    stream: bool = False,
    max_tokens: int = 4096,
    temperature: float = None,
    **kwargs,
) -> Union[dict, AsyncGenerator[str, None]]:
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic not installed — run: pip install anthropic")

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    # Extract system prompt if present
    system_prompt = None
    anthropic_msgs = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            anthropic_msgs.append({"role": m["role"], "content": m["content"]})

    params: dict = {
        "model": model,
        "messages": anthropic_msgs,
        "max_tokens": max_tokens,
    }
    if system_prompt:
        params["system"] = system_prompt
    if temperature is not None:
        params["temperature"] = temperature

    if stream:
        async def _stream() -> AsyncGenerator[str, None]:
            async with client.messages.stream(**params) as s:
                async for text in s.text_stream:
                    chunk = {
                        "choices": [{"delta": {"content": text}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        return _stream()

    response = await client.messages.create(**params)
    return {
        "id": response.id,
        "object": "chat.completion",
        "model": response.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response.content[0].text},
            "finish_reason": response.stop_reason,
        }],
        "usage": {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        },
    }


# ── OpenAI ────────────────────────────────────────────────────────────────────

async def call_openai(
    messages: List[Dict],
    model: str,
    stream: bool = False,
    max_tokens: int = 4096,
    temperature: float = None,
    **kwargs,
) -> Union[dict, AsyncGenerator[str, None]]:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai not installed — run: pip install openai")

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")

    client = AsyncOpenAI(api_key=settings.openai_api_key)

    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if temperature is not None:
        params["temperature"] = temperature

    if stream:
        async def _stream() -> AsyncGenerator[str, None]:
            response = await client.chat.completions.create(**params)
            async for chunk in response:
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        return _stream()

    response = await client.chat.completions.create(**params)
    return response.model_dump()


# ── Google Gemini ─────────────────────────────────────────────────────────────

async def call_gemini(
    messages: List[Dict],
    model: str,
    stream: bool = False,
    max_tokens: int = 4096,
    temperature: float = None,
    **kwargs,
) -> Union[dict, AsyncGenerator[str, None]]:
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY not set in .env")

    # Convert OpenAI-format messages to Gemini format
    contents = []
    for m in messages:
        role = "user" if m["role"] in ("user", "system") else "model"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"][0]["text"] += "\n" + m["content"]
        else:
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

    payload = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            **({"temperature": temperature} if temperature is not None else {}),
        },
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models"
        f"/{model}:generateContent"
    )

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            url,
            json=payload,
            params={"key": settings.google_api_key},
        )
        resp.raise_for_status()
        data = resp.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return {
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {},
    }
