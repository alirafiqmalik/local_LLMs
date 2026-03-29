"""task_type hint, direct Ollama name, system prompt, streaming."""
from _common import ExampleModels, client, get_example_models, router_meta, section


def run(models: ExampleModels | None = None) -> None:
    m = models or get_example_models()

    section("7. TASK TYPE HINT — bias routing toward code model")
    resp = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "explain how async/await works"}],
        extra_body={"task_type": "code"},
    )
    r = (getattr(resp, "model_extra", None) or {}).get("_router", {})
    print(f"  complexity : {r.get('complexity')}")
    print(f"  model used : {r.get('model_used')}  ← code path via task_type=code")
    print(f"  reply      :\n{resp.choices[0].message.content[:200]}")

    section(f"8. DIRECT MODEL NAME — {m.fast!r} (bypasses auto routing)")
    resp = client.chat.completions.create(
        model=m.fast,
        messages=[{"role": "user", "content": "what is 12 * 8?"}],
    )
    r = (getattr(resp, "model_extra", None) or {}).get("_router", {})
    print(f"  routing    : {r.get('routing_reason')}")
    print(f"  model used : {r.get('model_used')}")
    print(f"  reply      : {resp.choices[0].message.content[:80]}")

    section("9. SYSTEM PROMPT — custom persona")
    resp = client.chat.completions.create(
        model="auto",
        messages=[
            {
                "role": "system",
                "content": "You are a senior Python engineer. Be concise and opinionated.",
            },
            {"role": "user", "content": "list comprehensions or map()?"},
        ],
    )
    router_meta(resp)
    print(f"  reply      :\n{resp.choices[0].message.content[:300]}")

    section(f"10. STREAMING — {m.code!r}")
    print("  output:")
    print("  ", end="", flush=True)
    stream = client.chat.completions.create(
        model=m.code,
        messages=[
            {"role": "user", "content": "one-liner python to reverse a string"}
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


if __name__ == "__main__":
    run()
