"""Embeddings via router → Ollama (only if an embedding model is installed)."""
from _common import ExampleModels, client, get_example_models, section


def run(models: ExampleModels | None = None) -> None:
    m = models or get_example_models()

    if not m.embed:
        section("16. EMBEDDINGS — skipped (no embedding model installed)")
        print("  Install one of: nomic-embed-text  then re-run.")
        print("  Example: ollama pull nomic-embed-text")
        return

    section(f"16. EMBEDDINGS — {m.embed!r} via router → Ollama")
    emb = client.embeddings.create(
        model=m.embed,
        input="The quick brown fox jumps over the lazy dog",
    )
    vec = emb.data[0].embedding
    print(f"  model      : {emb.model}")
    print(f"  dimensions : {len(vec)}")
    print(f"  first 5    : {[round(v, 4) for v in vec[:5]]}")


if __name__ == "__main__":
    run()
