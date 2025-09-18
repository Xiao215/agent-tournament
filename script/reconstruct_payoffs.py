import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evolution.population_payoffs import PopulationPayoffs


def load_config(run_dir: Path) -> dict[str, Any]:
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def reconstruct_payoffs(run_dir: Path) -> Path:
    cfg = load_config(run_dir)
    agent_names = [a.get("name", "unknown") for a in (cfg.get("agents") or [])]
    discount = (cfg.get("mechanism") or {}).get("kwargs", {}).get("discount", 1.0)

    class _StubAgent:
        def __init__(self, name: str):
            self.name = name

    agents = [_StubAgent(name) for name in agent_names]
    name_to_agent = {agent.name: agent for agent in agents}

    pop = PopulationPayoffs(agents=agents, discount=discount)

    # Find the history file (*.jsonl)
    jsonl_files = sorted(run_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl history found in {run_dir}")
    hist_path = jsonl_files[0]

    rounds_count = 0
    for entry in iter_jsonl(hist_path):
        rounds = entry if (isinstance(entry, list) and entry and isinstance(entry[0], list)) else [entry]
        for round_moves in rounds:
            moves_objs = []
            seat_counts: defaultdict[str, int] = defaultdict(int)

            class _Move:
                __slots__ = ("label", "name", "points")

            for m in round_moves:
                agent = name_to_agent.get(m["name"])
                if agent is None:
                    continue
                seat_counts[agent.name] += 1
                seat_idx = seat_counts[agent.name]
                mo = _Move()
                mo.name = agent.name
                mo.label = f"{agent.name}#{seat_idx}"
                mo.points = float(m["points"])
                moves_objs.append(mo)
            pop.add_profile(moves_objs)
            rounds_count += 1

    record = pop.to_record()
    out_path = run_dir / "payoffs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(
        f"WROTE: {out_path} | profiles={len(record['profiles'])} rounds={rounds_count} discount={record['discount']}"
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Reconstruct payoffs.json from .jsonl histories")
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    reconstruct_payoffs(args.run_dir)


if __name__ == "__main__":
    main()
