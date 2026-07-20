from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_cities_land_uses() -> tuple[list[str], list[str]]:
    try:
        df = pd.read_csv(_PROJECT_ROOT / "data" / "parametros_ciudades.csv")
        if not df.empty and "city" in df.columns and df["city"].dropna().any():
            return sorted(df["city"].dropna().unique().tolist()), sorted(df["land_use"].dropna().unique().tolist())
    except Exception:
        pass
    return ["Bogotá", "Medellín", "Cali", "Cartagena"], ["Residencial", "Mixto"]


def _load_baseline() -> pd.DataFrame:
    try:
        df = pd.read_csv(_PROJECT_ROOT / "data" / "cronograma_planeado.csv")
        if not df.empty and "milestone" in df.columns:
            df["planned_date"] = pd.to_datetime(df["planned_date"], errors="coerce")
            df = df.dropna(subset=["milestone", "planned_date"])
            df["planned_date"] = df["planned_date"].dt.date
            return df
    except Exception:
        pass
    return pd.DataFrame(
        {
            "milestone": ["Cerramiento", "Excavación", "Cimentación", "Estructura", "Mampostería", "Instalaciones", "Acabados", "Entrega"],
            "planned_date": [date(2026, 1, 15), date(2026, 2, 10), date(2026, 3, 20), date(2026, 6, 30), date(2026, 8, 15), date(2026, 9, 30), date(2026, 11, 15), date(2026, 12, 20)],
        }
    )


_BASELINE_DF = _load_baseline()
CITIES, LAND_USES = _load_cities_land_uses()
MILESTONES = _BASELINE_DF["milestone"].tolist()
_PLANNED_DATES = _BASELINE_DF["planned_date"].tolist()
SOURCES = ["acta", "diario", "inventario", "foto"]
STATUSES = ["completed", "progress", "issue"]


def generate_projects(n: int = 100_000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    city = rng.choice(CITIES, size=n, replace=True)
    land_use = rng.choice(LAND_USES, size=n, replace=True)

    area_m2 = rng.lognormal(mean=np.log(1200.0), sigma=0.45, size=n)
    area_m2 = np.clip(area_m2, 200, 20000)

    floors_requested = rng.integers(4, 25, size=n)
    avg_unit_size_m2 = rng.normal(loc=65.0, scale=8.0, size=n)
    avg_unit_size_m2 = np.clip(avg_unit_size_m2, 35, 120)

    units = (area_m2 / avg_unit_size_m2 * rng.uniform(2.0, 5.5, size=n)).astype(int)
    units = np.clip(units, 10, 600)

    land_cost = rng.lognormal(mean=np.log(8.5e9), sigma=0.55, size=n)
    land_cost = np.clip(land_cost, 6e8, 8e10)

    df = pd.DataFrame(
        {
            "project_id": np.arange(1, n + 1, dtype=int),
            "city": city,
            "land_use": land_use,
            "area_m2": area_m2.round(1),
            "floors_requested": floors_requested.astype(int),
            "units": units.astype(int),
            "avg_unit_size_m2": avg_unit_size_m2.round(1),
            "land_cost": land_cost.round(0),
        }
    )
    return df


def generate_site_events(n: int = 100_000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, len(MILESTONES), size=n)
    milestone = np.array([MILESTONES[i] for i in idx])
    planned = np.array([_PLANNED_DATES[i] for i in idx], dtype=object)
    source = rng.choice(SOURCES, size=n, replace=True)
    status = rng.choice(STATUSES, size=n, replace=True, p=[0.55, 0.35, 0.10])

    offset = np.zeros(n, dtype=int)
    completed_mask = status == "completed"
    progress_mask = status == "progress"
    issue_mask = status == "issue"
    if completed_mask.any():
        offset[completed_mask] = rng.integers(-7, 31, size=completed_mask.sum())
    if progress_mask.any():
        offset[progress_mask] = rng.integers(-45, 11, size=progress_mask.sum())
    if issue_mask.any():
        offset[issue_mask] = rng.integers(-10, 16, size=issue_mask.sum())

    raw_dates = [planned[i] + timedelta(days=int(offset[i])) for i in range(n)]
    start = date(2026, 1, 1)
    end = date(2026, 12, 31)
    event_date = np.array([max(start, min(end, d)) for d in raw_dates], dtype=object)

    detail_templates = {
        "completed": "Evidencia: hito finalizado.",
        "progress": "Evidencia: avance parcial reportado.",
        "issue": "Alerta: novedad / restricción en el hito.",
    }
    detail = np.array([detail_templates[s] for s in status], dtype=object)

    df = pd.DataFrame(
        {
            "event_id": np.arange(1, n + 1, dtype=int),
            "milestone": milestone,
            "event_date": pd.to_datetime(event_date),
            "status": status,
            "source": source,
            "detail": detail,
        }
    )

    df = df.sort_values(["milestone", "event_date", "status"], ascending=[True, True, True]).reset_index(drop=True)
    df["event_id"] = np.arange(1, n + 1, dtype=int)
    df["event_date"] = df["event_date"].dt.date

    return df
