"""
LLM Router — Usage Examples
Run: ./llm examples
     OR: .venv/bin/python examples/examples.py
Requires router + Ollama running: ./llm start
HF engine examples require: ./llm start --hf
"""
import httpx
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


def router_meta(resp):
    r = resp.model_extra.get("_router", {})
    print(f"  complexity : {r.get('complexity')}")
    print(f"  model used : {r.get('model_used')}")
    print(f"  provider   : {r.get('provider')}")
    print(f"  latency    : {r.get('latency_ms')}ms")


# ═══════════════════════════════════════════════════════
#  OLLAMA BACKEND — Auto-routing
# ═══════════════════════════════════════════════════════

section("1. AUTO ROUTING — trivial → gemma3:4b")
resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "hi"}],
)
router_meta(resp)
print(f"  reply      : {resp.choices[0].message.content[:80]}")


section("2. AUTO ROUTING — code → qwen2.5-coder")
resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "write a python function to flatten a nested list"}],
)
router_meta(resp)
print(f"  reply      :\n{resp.choices[0].message.content[:300]}")


section("3. AUTO ROUTING — complex → mistral:7b")
resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "refactor this code using a design pattern and explain tradeoffs: def process(items): result=[]; [result.append(x*2) for x in items if x>0]; return result"}],
)
router_meta(resp)
print(f"  reply      :\n{resp.choices[0].message.content[:300]}")


# ═══════════════════════════════════════════════════════
#  OLLAMA BACKEND — Virtual routing tags
# ═══════════════════════════════════════════════════════

section("4. FORCE local:fast  (always gemma3:4b)")
resp = client.chat.completions.create(
    model="local:fast",
    messages=[{"role": "user", "content": "what is a REST API?"}],
)
r = resp.model_extra.get("_router", {})
print(f"  model used : {r.get('model_used')}")
print(f"  reply      : {resp.choices[0].message.content[:120]}")


section("5. FORCE local:code  (always qwen2.5-coder)")
resp = client.chat.completions.create(
    model="local:code",
    messages=[{"role": "user", "content": "write a rust function that reads a file line by line"}],
)
r = resp.model_extra.get("_router", {})
print(f"  model used : {r.get('model_used')}")
print(f"  reply      :\n{resp.choices[0].message.content[:300]}")


section("6. FORCE local:general  (always mistral:7b)")
resp = client.chat.completions.create(
    model="local:general",
    messages=[{"role": "user", "content": "pros and cons of microservices vs monolith?"}],
)
r = resp.model_extra.get("_router", {})
print(f"  model used : {r.get('model_used')}")
print(f"  reply      :\n{resp.choices[0].message.content[:300]}")


# ═══════════════════════════════════════════════════════
#  OLLAMA BACKEND — Advanced options
# ═══════════════════════════════════════════════════════

section("7. TASK TYPE HINT — bias routing toward code model")
resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "explain how async/await works"}],
    extra_body={"task_type": "code"},
)
r = resp.model_extra.get("_router", {})
print(f"  complexity : {r.get('complexity')}")
print(f"  model used : {r.get('model_used')}  ← code model via task_type hint")
print(f"  reply      :\n{resp.choices[0].message.content[:200]}")


section("8. DIRECT MODEL NAME — bypass routing entirely")
resp = client.chat.completions.create(
    model="gemma3:4b",
    messages=[{"role": "user", "content": "what is 12 * 8?"}],
)
r = resp.model_extra.get("_router", {})
print(f"  model used : {r.get('model_used')}")
print(f"  reply      : {resp.choices[0].message.content[:80]}")


section("9. SYSTEM PROMPT — custom persona")
resp = client.chat.completions.create(
    model="auto",
    messages=[
        {"role": "system", "content": "You are a senior Python engineer. Be concise and opinionated."},
        {"role": "user",   "content": "list comprehensions or map()?"},
    ],
)
r = resp.model_extra.get("_router", {})
print(f"  model used : {r.get('model_used')}")
print(f"  reply      :\n{resp.choices[0].message.content[:300]}")


section("10. STREAMING — token-by-token output")
print("  model: local:code  |  output:")
print("  ", end="", flush=True)
stream = client.chat.completions.create(
    model="local:code",
    messages=[{"role": "user", "content": "one-liner python to reverse a string"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()


# ═══════════════════════════════════════════════════════
#  HF ENGINE BACKEND
#  Requires: ./llm start --hf
#  First call downloads + loads the model (~30-120s)
# ═══════════════════════════════════════════════════════

section("11. HF ENGINE — check if running")
try:
    hf_health = httpx.get("http://localhost:8001/health", timeout=3).json()
    hf_ok = hf_health.get("status") == "ok"
    print(f"  hf engine  : {'✓ running' if hf_ok else '✗ not running'}")
    print(f"  model loaded: {hf_health.get('model_id') or 'none'}")
    print(f"  backend    : {hf_health.get('backend') or '—'}")
except Exception:
    hf_ok = False
    print("  hf engine  : ✗ not running — skipping HF examples")
    print("  start with : ./llm start --hf")

if hf_ok:
    section("12. HF ENGINE — load any HuggingFace model (4-bit auto-quant)")
    print("  NOTE: First call downloads + loads model. May take 1-3 min.")
    print("  Using: Qwen/Qwen2.5-0.5B-Instruct (tiny, fast to download)")
    try:
        resp = client.chat.completions.create(
            model="hf:Qwen/Qwen2.5-0.5B-Instruct",
            messages=[{"role": "user", "content": "write hello world in python"}],
        )
        r = resp.model_extra.get("_router", {})
        print(f"  model used : {r.get('model_used')}")
        print(f"  provider   : {r.get('provider')}")
        print(f"  latency    : {r.get('latency_ms')}ms")
        print(f"  reply      :\n{resp.choices[0].message.content[:300]}")
    except Exception as e:
        print(f"  error: {e}")

    section("13. HF ENGINE — direct load via engine API")
    print("  Explicitly load a model with chosen quantization:")
    try:
        load_resp = httpx.post(
            "http://localhost:8001/v1/models/load",
            json={"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "quantization": "4bit"},
            timeout=180,
        ).json()
        print(f"  status     : {load_resp.get('status')}")
        print(f"  backend    : {load_resp.get('backend')}")
        print(f"  vram used  : {load_resp.get('vram_used_gb')} GB")
    except Exception as e:
        print(f"  error: {e}")

    section("14. HF ENGINE — unload (free VRAM)")
    unload_resp = httpx.post("http://localhost:8001/v1/models/unload", timeout=10).json()
    print(f"  status     : {unload_resp.get('status')}")
    print(f"  VRAM freed for Ollama models")

    section("15. HF ENGINE — local GGUF file")
    print("  Place a .gguf file in models/huggingface/ then call:")
    print('  model="hf:my-model.gguf"  ← looked up in models/huggingface/')
    print("  (skipping — no .gguf file present in this example)")


# ═══════════════════════════════════════════════════════
#  EMBEDDINGS
# ═══════════════════════════════════════════════════════

section("16. EMBEDDINGS — nomic-embed-text via Ollama")
emb = client.embeddings.create(
    model="nomic-embed-text",
    input="The quick brown fox jumps over the lazy dog",
)
vec = emb.data[0].embedding
print(f"  model      : {emb.model}")
print(f"  dimensions : {len(vec)}")
print(f"  first 5    : {[round(v, 4) for v in vec[:5]]}")


# ═══════════════════════════════════════════════════════
#  ROUTER STATUS
# ═══════════════════════════════════════════════════════

section("17. ROUTER STATUS — full system snapshot")
status = httpx.get("http://localhost:8000/api/status").json()
print(f"  uptime     : {status['uptime_s']}s")
print(f"  ollama     : {'✓' if status['ollama']['ok'] else '✗'}  ({len(status['ollama']['models'])} models)")
print(f"  hf engine  : {'✓' if status['hf_engine']['ok'] else '✗ (not running)'}")
print(f"  total reqs : {status['stats']['total_requests']}")
print(f"  by model   :\n{json.dumps(status['stats']['by_model'], indent=6)}")
print(f"  by complexity:\n{json.dumps(status['stats']['by_complexity'], indent=6)}")

print(f"\n{'─'*60}")
print("  All examples complete.")
print('─'*60)
