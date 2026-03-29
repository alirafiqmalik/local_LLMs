"""Auto complexity routing (model=auto) — Ollama backend."""
from _common import ExampleModels, client, get_example_models, router_meta, section


def run(models: ExampleModels | None = None) -> None:
    m = models or get_example_models()

    section(f"1. AUTO ROUTING — trivial → {m.routing_table.get('trivial', m.fast)} (routing table)")
    resp = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "hi"}],
    )
    router_meta(resp)
    print(f"  reply      : {resp.choices[0].message.content[:80]}")

    section(f"2. AUTO ROUTING — code → {m.routing_table.get('medium_code', m.code)} (routing table)")
    resp = client.chat.completions.create(
        model="auto",
        messages=[
            {
                "role": "user",
                "content": "write a python function to flatten a nested list",
            }
        ],
    )
    router_meta(resp)
    print(f"  reply      :\n{resp.choices[0].message.content[:300]}")

    complex_model = m.routing_table.get("complex", "")
    if complex_model in m.installed:
        section(f"3. AUTO ROUTING — complex → {complex_model} (routing table)")
        resp = client.chat.completions.create(
            model="auto",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "refactor this code using a design pattern and explain tradeoffs: "
                        "def process(items): result=[]; "
                        "[result.append(x*2) for x in items if x>0]; return result"
                    ),
                }
            ],
        )
    else:
        section(
            f"3. COMPLEX PROMPT — table wants {complex_model!r} (not installed); "
            f"using {m.general!r} directly"
        )
        print(
            "  (Pull the table model or set MODEL_COMPLEX in .env to an installed id "
            "to exercise full auto routing.)"
        )
        resp = client.chat.completions.create(
            model=m.general,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "refactor this code using a design pattern and explain tradeoffs: "
                        "def process(items): result=[]; "
                        "[result.append(x*2) for x in items if x>0]; return result"
                    ),
                }
            ],
        )
    router_meta(resp)
    print(f"  reply      :\n{resp.choices[0].message.content[:300]}")


if __name__ == "__main__":
    run()
