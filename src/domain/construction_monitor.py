from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class BaselineMilestone:
    milestone: str
    planned_date: date
    weight: float


def compute_progress(baseline: pd.DataFrame, events: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Return milestone-level progress comparison.

    baseline columns: milestone, planned_date, weight
    events columns: milestone, event_date, status
    """

    b = baseline.copy()
    e = events.copy()

    b["planned_date"] = pd.to_datetime(b["planned_date"]).dt.date
    e["event_date"] = pd.to_datetime(e["event_date"]).dt.date

    e = e[e["event_date"] <= as_of]

    completed = (
        e.sort_values(["milestone", "event_date"])
        .groupby("milestone", as_index=False)
        .tail(1)
        .set_index("milestone")
    )

    b["completed"] = b["milestone"].map(lambda m: m in completed.index)
    b["actual_date"] = b["milestone"].map(lambda m: completed.loc[m, "event_date"] if m in completed.index else None)

    b["delay_days"] = b.apply(
        lambda r: (r["actual_date"] - r["planned_date"]).days if r["completed"] and r["actual_date"] else None,
        axis=1,
    )

    b["planned_completed"] = b["planned_date"].map(lambda d: d <= as_of)

    b["risk"] = b.apply(
        lambda r: "Atrasado" if (r["planned_completed"] and not r["completed"]) else ("OK" if r["completed"] else "Pendiente"),
        axis=1,
    )

    total_weight = float(b["weight"].sum()) if float(b["weight"].sum()) > 0 else 1.0
    b["weight_norm"] = b["weight"] / total_weight

    planned_progress = float(b.loc[b["planned_completed"], "weight_norm"].sum())
    actual_progress = float(b.loc[b["completed"], "weight_norm"].sum())

    summary = pd.DataFrame(
        {
            "metric": ["planned_progress", "actual_progress", "delta"],
            "value": [planned_progress, actual_progress, actual_progress - planned_progress],
        }
    )

    return b, summary
