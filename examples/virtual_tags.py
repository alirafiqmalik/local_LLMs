"""Direct Ollama names for fast / code / general (only installed tags)."""
from _common import ExampleModels, client, get_example_models, section


def run(models: ExampleModels | None = None) -> None:
    m = models or get_example_models()

    section(f"4. FAST MODEL — {m.fast!r} (same role as local:fast)")
    resp = client.chat.completions.create(
        model=m.fast,
        messages=[{"role": "user", "content": "what is a REST API?"}],
    )
    r = (getattr(resp, "model_extra", None) or {}).get("_router", {})
    print(f"  model used : {r.get('model_used')}")
    print(f"  reply      : {resp.choices[0].message.content[:120]}")

    section(f"5. CODE MODEL — {m.code!r} (same role as local:code)")
    resp = client.chat.completions.create(
        model=m.code,
        messages=[
            {
                "role": "user",
                "content": "write a rust function that reads a file line by line",
            }
        ],
    )
    r = (getattr(resp, "model_extra", None) or {}).get("_router", {})
    print(f"  model used : {r.get('model_used')}")
    print(f"  reply      :\n{resp.choices[0].message.content[:300]}")

    section(f"6. GENERAL MODEL — {m.general!r} (same role as local:general)")
    resp = client.chat.completions.create(
        model=m.general,
        messages=[
            {
                "role": "user",
                "content": "pros and cons of microservices vs monolith?",
            }
        ],
    )
    r = (getattr(resp, "model_extra", None) or {}).get("_router", {})
    print(f"  model used : {r.get('model_used')}")
    print(f"  reply      :\n{resp.choices[0].message.content[:300]}")


if __name__ == "__main__":
    run()
