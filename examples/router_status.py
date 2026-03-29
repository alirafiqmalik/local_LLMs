"""GET /api/status — router uptime, backends, request stats."""
import httpx

from _common import ROUTER_API, ExampleModels, pretty_json, section


def run(models: ExampleModels | None = None) -> None:
    section("17. ROUTER STATUS — full system snapshot")
    status = httpx.get(f"{ROUTER_API}/api/status").json()
    print(f"  uptime     : {status['uptime_s']}s")
    print(
        f"  ollama     : {'✓' if status['ollama']['ok'] else '✗'}  "
        f"({len(status['ollama']['models'])} models)"
    )
    print(
        f"  hf engine  : {'✓' if status['hf_engine']['ok'] else '✗ (not running)'}"
    )
    st = status["stats"]
    print(f"  total reqs : {st['total_requests']}")
    print(f"  errors     : {st.get('errors', 0)}")
    print(f"  by model   :\n{pretty_json(st['by_model'])}")
    print(f"  by complexity:\n{pretty_json(st['by_complexity'])}")


if __name__ == "__main__":
    run()
