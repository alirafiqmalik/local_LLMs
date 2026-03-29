"""
LLM Router — Main FastAPI Application

Exposes:
  GET  /                          → Dashboard UI
  GET  /api/status                → Router health + stats
  GET  /api/models                → All Ollama installs + routing table
  POST /v1/chat/completions       → OpenAI-compatible routed completions
  POST /v1/embeddings             → Forwarded to Ollama nomic-embed-text
  WS   /ws/metrics                → Real-time system metrics stream
"""
import asyncio
import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import List, Optional, Any, Union

import httpx
import psutil
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from config import settings
from classifier import classify, Complexity
from models import resolve_model, LOCAL_MODELS
from metrics import get_system_metrics


# ── State ─────────────────────────────────────────────────────────────────────

request_log: deque = deque(maxlen=200)

stats: dict = {
    "total_requests": 0,
    "by_model": {},
    "by_complexity": {},
    "errors": 0,
    "start_time": time.time(),
}

ws_clients: List[WebSocket] = []

DASHBOARD_PATH = Path(__file__).parent.parent / "dashboard" / "index.html"


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Optional routing hints
    task_type: Optional[str] = None   # "code" | "analysis" | "general"


# ── URLs ──────────────────────────────────────────────────────────────────────

OLLAMA_URL    = settings.ollama_base_url
HF_ENGINE_URL = settings.hf_engine_url


# ── VRAM coordinator ──────────────────────────────────────────────────────────
# Ensures only one model is in GPU VRAM at a time.
# Both engines idle at ~0 VRAM when no model is loaded.

async def _ollama_unload_all():
    """Tell Ollama to unload any running model (keep_alive=0)."""
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            ps = await c.get(f"{OLLAMA_URL}/api/ps")
            for m in ps.json().get("models", []):
                await c.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={"model": m["name"], "messages": [], "keep_alive": 0},
                )
    except Exception:
        pass  # Ollama may not be running


async def _hf_unload():
    """Tell HF engine to release its model from VRAM."""
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(f"{HF_ENGINE_URL}/v1/models/unload")
    except Exception:
        pass  # HF engine may not be running


async def prepare_vram(target_backend: str):
    """
    Free VRAM from the other backend before a request.
    Called automatically before every inference request.
    """
    if target_backend == "hf_engine":
        await _ollama_unload_all()
    elif target_backend == "ollama":
        await _hf_unload()


# ── Ollama proxy ──────────────────────────────────────────────────────────────

async def call_ollama(
    model: str,
    messages: list,
    stream: bool = False,
    temperature: float = None,
    max_tokens: int = None,
) -> Any:
    payload: dict = {"model": model, "messages": messages, "stream": stream}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if stream:
        async def _stream():
            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_URL}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"{line}\n\n"
        return _stream()

    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


# ── HF Engine proxy ───────────────────────────────────────────────────────────

async def call_hf_engine(
    model: str,
    messages: list,
    stream: bool = False,
    temperature: float = None,
    max_tokens: int = None,
) -> Any:
    payload: dict = {
        "model":       model,
        "messages":    messages,
        "stream":      stream,
        "temperature": temperature if temperature is not None else 0.7,
        "max_tokens":  max_tokens or 2048,
    }

    if stream:
        async def _stream():
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream(
                    "POST",
                    f"{HF_ENGINE_URL}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line:
                            yield f"{line}\n\n"
        return _stream()

    # Non-streaming — HF model load can take 30-120s on first request
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            f"{HF_ENGINE_URL}/v1/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


# ── Logging + broadcasting ────────────────────────────────────────────────────

def _record(req_id, model, provider, complexity, reason, duration_ms, error=None):
    entry = {
        "id": req_id,
        "time": time.strftime("%H:%M:%S"),
        "model": model,
        "provider": provider,
        "complexity": complexity,
        "reason": reason,
        "duration_ms": round(duration_ms),
        "error": error,
        "ts": time.time(),
    }
    request_log.appendleft(entry)
    stats["total_requests"] += 1
    stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
    stats["by_complexity"][complexity] = stats["by_complexity"].get(complexity, 0) + 1
    if error:
        stats["errors"] += 1

    asyncio.create_task(_broadcast())


async def _broadcast():
    if not ws_clients:
        return
    payload = json.dumps({
        "type": "update",
        "system": get_system_metrics(),
        "stats": stats,
        "recent": list(request_log)[:30],
    })
    dead = []
    for ws in list(ws_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ── App lifecycle ─────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up CPU percent baseline
    psutil.cpu_percent(percpu=True, interval=None)

    async def _metrics_loop():
        while True:
            await _broadcast()
            await asyncio.sleep(2)

    task = asyncio.create_task(_metrics_loop())
    yield
    task.cancel()


app = FastAPI(title="LLM Router", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    if DASHBOARD_PATH.exists():
        return HTMLResponse(DASHBOARD_PATH.read_text())
    return HTMLResponse("<h1>Dashboard not found</h1><p>Ensure dashboard/index.html exists.</p>")


@app.get("/api/status")
async def status():
    # Quick Ollama health check
    ollama_ok = False
    ollama_models = []
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                ollama_ok = True
                ollama_models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    # Quick HF engine health check
    hf_ok = False
    hf_status = {}
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{HF_ENGINE_URL}/health")
            if r.status_code == 200:
                hf_ok = True
                hf_status = r.json()
    except Exception:
        pass

    return {
        "status": "ok",
        "uptime_s": round(time.time() - stats["start_time"]),
        "ollama": {"url": OLLAMA_URL, "ok": ollama_ok, "models": ollama_models},
        "hf_engine": {"url": HF_ENGINE_URL, "ok": hf_ok, **hf_status},
        "stats": stats,
    }


@app.get("/api/models")
async def models_list():
    def _registry_match(ollama_name: str):
        for k, v in LOCAL_MODELS.items():
            if k in ollama_name or ollama_name in k:
                return k, v
        return None, None

    local_installed: List[dict] = []
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            if r.status_code == 200:
                raw = r.json().get("models") or []
                rows = []
                for m in raw:
                    name = m.get("name")
                    if not name:
                        continue
                    item: dict = {
                        "id": name,
                        "size": m.get("size"),
                        "modified_at": m.get("modified_at"),
                        "digest": m.get("digest"),
                    }
                    _rk, rv = _registry_match(name)
                    if rv is not None:
                        item["vram_gb"] = rv.vram_gb
                        item["description"] = rv.description
                        item["recommended_for"] = rv.recommended_for
                    rows.append(item)
                local_installed = sorted(rows, key=lambda x: x["id"].lower())
    except Exception:
        pass

    return {
        "local_installed": local_installed,
        "routing_table": {
            "trivial":        settings.model_trivial,
            "simple":         settings.model_simple,
            "medium_code":    settings.model_medium_code,
            "medium_general": settings.model_medium_general,
            "complex":        settings.model_complex,
            "expert":         settings.model_expert,
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    req_id = str(uuid.uuid4())[:8]
    start = time.time()

    messages = [m.model_dump() for m in req.messages]

    # 1. Classify complexity
    complexity, classify_reason = classify(messages)

    # 2. Resolve model + provider
    provider, model, routing_reason = resolve_model(
        requested_model=req.model,
        complexity=complexity,
        messages=messages,
        task_type=req.task_type or "general",
    )

    error = None
    try:
        # 3. Free VRAM on the other backend before loading
        await prepare_vram(provider)

        # 4. Call the right backend
        if provider == "ollama":
            result = await call_ollama(
                model=model,
                messages=messages,
                stream=req.stream,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
        elif provider == "hf_engine":
            result = await call_hf_engine(
                model=model,
                messages=messages,
                stream=req.stream,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
        else:
            raise HTTPException(500, f"Unknown provider: {provider}")

        duration_ms = (time.time() - start) * 1000
        _record(req_id, model, provider, complexity.value, routing_reason, duration_ms)

        if req.stream:
            return StreamingResponse(result, media_type="text/event-stream")

        # Attach routing metadata
        if isinstance(result, dict):
            result["_router"] = {
                "req_id": req_id,
                "complexity": complexity.value,
                "classify_reason": classify_reason,
                "routing_reason": routing_reason,
                "model_used": model,
                "provider": provider,
                "latency_ms": round(duration_ms),
            }
        return JSONResponse(result)

    except Exception as exc:
        error = str(exc)
        duration_ms = (time.time() - start) * 1000
        _record(req_id, model, provider, complexity.value, routing_reason, duration_ms, error)
        raise HTTPException(status_code=502, detail=error)


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """Pass-through to Ollama embeddings endpoint."""
    body = await request.json()
    body.setdefault("model", "nomic-embed-text")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(f"{OLLAMA_URL}/v1/embeddings", json=body)
        return JSONResponse(resp.json())


@app.websocket("/ws/metrics")
async def metrics_ws(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    # Send initial snapshot
    await ws.send_text(json.dumps({
        "type": "init",
        "system": get_system_metrics(),
        "stats": stats,
        "recent": list(request_log)[:30],
    }))
    try:
        while True:
            await ws.receive_text()   # keep-alive pings
    except WebSocketDisconnect:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )
