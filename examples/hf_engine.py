"""HF engine on port 8001 — requires ./llm start --hf."""
import httpx

from _common import HF_ENGINE, ExampleModels, client, router_meta, section


def run(models: ExampleModels | None = None) -> None:
    section("11. HF ENGINE — health check")
    hf_ok = False
    try:
        hf_health = httpx.get(f"{HF_ENGINE}/health", timeout=3).json()
        hf_ok = hf_health.get("status") == "ok"
        print(f"  hf engine  : {'✓ running' if hf_ok else '✗ not running'}")
        print(f"  model loaded: {hf_health.get('model_id') or 'none'}")
        print(f"  backend    : {hf_health.get('backend') or '—'}")
    except Exception:
        print("  hf engine  : ✗ not running — skipping HF examples")
        print("  start with : ./llm start --hf")

    if not hf_ok:
        return

    section("12. HF ENGINE — hf:… via router (4-bit load on first use)")
    print("  NOTE: First call downloads + loads model. May take 1–3 min.")
    print("  Using: Qwen/Qwen2.5-0.5B-Instruct (small, quick download)")
    try:
        resp = client.chat.completions.create(
            model="hf:Qwen/Qwen2.5-0.5B-Instruct",
            messages=[{"role": "user", "content": "write hello world in python"}],
        )
        router_meta(resp)
        print(f"  reply      :\n{resp.choices[0].message.content[:300]}")
    except Exception as e:
        print(f"  error: {e}")

    section("13. HF ENGINE — explicit load via engine API")
    print("  Load with chosen quantization:")
    try:
        load_resp = httpx.post(
            f"{HF_ENGINE}/v1/models/load",
            json={"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "quantization": "4bit"},
            timeout=180,
        ).json()
        print(f"  status     : {load_resp.get('status')}")
        print(f"  backend    : {load_resp.get('backend')}")
        print(f"  vram used  : {load_resp.get('vram_used_gb')} GB")
    except Exception as e:
        print(f"  error: {e}")

    section("14. HF ENGINE — unload (free VRAM for Ollama)")
    try:
        unload_resp = httpx.post(f"{HF_ENGINE}/v1/models/unload", timeout=10).json()
        print(f"  status     : {unload_resp.get('status')}")
    except Exception as e:
        print(f"  error: {e}")

    section("15. HF ENGINE — local GGUF")
    print("  Put a .gguf in models/huggingface/ then use model=\"hf:my-model.gguf\"")
    print("  (skipped here — no bundled GGUF)")


if __name__ == "__main__":
    run()
