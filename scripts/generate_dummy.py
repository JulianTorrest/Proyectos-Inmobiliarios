from __future__ import annotations

import argparse
from pathlib import Path

import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.generate import generate_projects, generate_site_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", type=int, default=100_000)
    parser.add_argument("--events", type=int, default=100_000)
    parser.add_argument("--seed-projects", type=int, default=7)
    parser.add_argument("--seed-events", type=int, default=11)
    parser.add_argument("--out-dir", type=str, default="data")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    projects = generate_projects(n=args.projects, seed=args.seed_projects)
    events = generate_site_events(n=args.events, seed=args.seed_events)

    projects_path = out_dir / "projects_large.csv"
    events_path = out_dir / "site_events_large.csv"

    projects.to_csv(projects_path, index=False)
    events.to_csv(events_path, index=False)

    print(f"Wrote: {projects_path} ({len(projects):,} rows)")
    print(f"Wrote: {events_path} ({len(events):,} rows)")


if __name__ == "__main__":
    main()
