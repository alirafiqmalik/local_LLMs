"""
LLM Router — run all usage examples (or one module).

  ./llm examples
  .venv/bin/python examples/examples.py
  .venv/bin/python examples/examples.py auto_routing

Requires router + Ollama: ./llm start
Chat examples use only models reported in GET /api/models → local_installed.
HF sections require: ./llm start --hf
"""
import sys

import advanced_chat
import auto_routing
import embeddings
import hf_engine
import router_status
import virtual_tags
from _common import get_example_models

_RUNNERS = {
    "auto_routing": auto_routing.run,
    "virtual_tags": virtual_tags.run,
    "advanced_chat": advanced_chat.run,
    "advanced": advanced_chat.run,
    "hf_engine": hf_engine.run,
    "hf": hf_engine.run,
    "embeddings": embeddings.run,
    "router_status": router_status.run,
    "status": router_status.run,
}

_NO_MODEL_RESOLUTION = frozenset({"router_status", "status", "hf_engine", "hf"})


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if argv:
        name = argv[0].lower().replace(".py", "")
        fn = _RUNNERS.get(name)
        if not fn:
            known = ", ".join(sorted(set(_RUNNERS)))
            print(f"Unknown example {argv[0]!r}. Choose one of: {known}")
            sys.exit(1)
        if name in _NO_MODEL_RESOLUTION:
            fn()
        else:
            fn(get_example_models())
        return

    models = get_example_models()
    for fn in (
        auto_routing.run,
        virtual_tags.run,
        advanced_chat.run,
        hf_engine.run,
        embeddings.run,
        router_status.run,
    ):
        fn(models)
    print(f"\n{'─'*60}")
    print("  All examples complete.")
    print("─" * 60)


if __name__ == "__main__":
    main()
