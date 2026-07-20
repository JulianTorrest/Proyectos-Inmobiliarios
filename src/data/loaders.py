from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.generate import generate_projects, generate_site_events


def project_root() -> Path:
    # app.py lives at repo root
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_root() / "data"


_CSV_CACHE: dict[str, pd.DataFrame] = {}


def _cached_csv(name: str, fallback: str | None = None) -> pd.DataFrame:
    if name not in _CSV_CACHE:
        path = data_dir() / name
        if not path.exists() and fallback:
            path = data_dir() / fallback
        _CSV_CACHE[name] = pd.read_csv(path)
    return _CSV_CACHE[name]


def load_normative_rules() -> pd.DataFrame:
    return _cached_csv("normativa_ciudades.csv", "normative_rules_co.csv")


def load_market_assumptions() -> pd.DataFrame:
    return _cached_csv("parametros_ciudades.csv", "market_assumptions_co.csv")


def load_baseline_schedule() -> pd.DataFrame:
    return _cached_csv("cronograma_planeado.csv", "baseline_schedule_dummy.csv")


def load_site_events() -> pd.DataFrame:
    return _cached_csv("site_events_large.csv", "site_events_dummy.csv")


_PROJECTS_CACHE: dict[tuple[int, int], pd.DataFrame] = {}
_EVENTS_CACHE: dict[tuple[int, int], pd.DataFrame] = {}


def load_projects_large(n: int = 100_000, seed: int = 7) -> pd.DataFrame:
    key = (int(n), int(seed))
    if key not in _PROJECTS_CACHE:
        _PROJECTS_CACHE[key] = generate_projects(n=n, seed=seed)
    return _PROJECTS_CACHE[key]


def load_site_events_large(n: int = 100_000, seed: int = 11) -> pd.DataFrame:
    key = (int(n), int(seed))
    if key not in _EVENTS_CACHE:
        _EVENTS_CACHE[key] = generate_site_events(n=n, seed=seed)
    return _EVENTS_CACHE[key]
