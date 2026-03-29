"""
HF Engine — FastAPI inference server  (port 8001)

OpenAI-compatible API backed by HuggingFace transformers / llama-cpp-python.

Endpoints:
  GET  /health                 → engine status + loaded model info
  GET  /v1/models              → list loaded model
  POST /v1/models/load         → {model_id, quantization} — load into VRAM
  POST /v1/models/unload       → free VRAM
  POST /v1/chat/completions    → OpenAI-compatible inference (stream supported)
"""
import asyncio
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import loader

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # local_LLMs/

app = FastAPI(title="HF Engine", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single-worker executor — one inference at a time (VRAM constraint)
_executor = ThreadPoolExecutor(max_workers=1)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048


class LoadRequest(BaseModel):
    model_id: str
    quantization: str = "4bit"   # "4bit" | "8bit" | "none" | "auto"


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _run_sync(fn):
    """Run a blocking function in the thread pool without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn)


def _make_chunk(content: str, finish: bool = False) -> str:
    chunk = {
        "id": f"hf-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "model": loader.status().get("model_id", "unknown"),
        "choices": [{
            "delta": {"content": content} if not finish else {},
            "finish_reason": "stop" if finish else None,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    s = loader.status()
    return {
        "status": "ok",
        "engine": "hf",
        "port": int(os.environ.get("HF_ENGINE_PORT", 8001)),
        **s,
    }


@app.get("/v1/models")
async def list_models():
    s = loader.status()
    data = []
    if s["loaded"]:
        data.append({
            "id":       s["model_id"],
            "object":   "model",
            "owned_by": "huggingface",
            "backend":  s["backend"],
        })
    return {"object": "list", "data": data}


@app.post("/v1/models/load")
async def load_model(req: LoadRequest):
    try:
        result = await _run_sync(lambda: loader.load(req.model_id, req.quantization))
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"Load failed: {e}")


@app.post("/v1/models/unload")
async def unload_model():
    return loader.unload()


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    messages = [m.model_dump() for m in req.messages]
    s = loader.status()

    # Auto-load if no model or wrong model is loaded
    if not s["loaded"] or s["model_id"] != req.model:
        try:
            await _run_sync(lambda: loader.load(req.model, "4bit"))
        except Exception as e:
            raise HTTPException(502, f"Failed to load '{req.model}': {e}")

    temperature = req.temperature if req.temperature is not None else 0.7
    max_tokens  = req.max_tokens or 2048
    model_id    = loader.status()["model_id"]

    if req.stream:
        async def _stream_gen():
            # Run blocking generate in executor, get streamer back
            streamer = await _run_sync(
                lambda: loader.generate(messages, max_tokens, temperature, stream=True)
            )
            # Stream tokens — streamer may be a TextIteratorStreamer or llama-cpp generator
            for item in streamer:
                # llama-cpp returns dicts; transformers streamer returns strings
                if isinstance(item, dict):
                    token = item.get("choices", [{}])[0].get("delta", {}).get("content", "")
                else:
                    token = item
                if token:
                    yield _make_chunk(token)
            yield _make_chunk("", finish=True)
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream_gen(), media_type="text/event-stream")

    try:
        text = await _run_sync(
            lambda: loader.generate(messages, max_tokens, temperature, stream=False)
        )
    except Exception as e:
        raise HTTPException(502, f"Inference error: {e}")

    return JSONResponse({
        "id":      f"hf-{uuid.uuid4().hex[:8]}",
        "object":  "chat.completion",
        "model":   model_id,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {},
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("HF_ENGINE_PORT", 8001))
    print(f"HF Engine starting on port {port}")
    print(f"  Models dir : {loader.HF_MODELS_DIR}")
    print(f"  Base dir   : {BASE_DIR}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
