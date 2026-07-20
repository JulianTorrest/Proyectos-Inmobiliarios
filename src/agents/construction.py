from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from src.domain.construction_monitor import compute_progress
from src.llm.client import MultiProviderLLM
from src.llm.providers import LLMProviderConfig


@dataclass(frozen=True)
class ConstructionMonitorOutputs:
    milestones: pd.DataFrame
    summary: pd.DataFrame
    alert_report: str


def run_construction_monitor(
    baseline_df: pd.DataFrame,
    events_df: pd.DataFrame,
    as_of: date,
    llm: MultiProviderLLM,
    provider: LLMProviderConfig,
    model: Optional[str] = None,
    use_llm: bool = True,
) -> ConstructionMonitorOutputs:
    milestones, summary = compute_progress(baseline_df, events_df, as_of)

    atrasados = milestones[milestones["risk"] == "Atrasado"]["milestone"].tolist()
    delta = float(summary.loc[summary["metric"] == "delta", "value"].iloc[0])

    alert_report = ""

    if use_llm:
        system = (
            "Eres un PMO senior de obra (Colombia). "
            "Debes generar alertas accionables, con foco en cuellos de botella y mitigaciones. "
            "No inventes datos que no estén en los inputs."
        )
        user = (
            f"Fecha de corte: {as_of.isoformat()}\n\n"
            "Baseline (hitos y fechas):\n"
            f"{baseline_df.to_csv(index=False)}\n\n"
            "Eventos reales (actas/diarios):\n"
            f"{events_df.to_csv(index=False)}\n\n"
            "Resultado comparativo (por hito):\n"
            f"{milestones[['milestone','planned_date','completed','actual_date','risk','delay_days']].to_csv(index=False)}\n\n"
            f"Delta de avance (actual - planeado): {delta:.2%}\n"
            f"Hitos atrasados: {', '.join(atrasados) if atrasados else 'N/A'}\n\n"
            "Entrega: 1) Semáforo general (verde/amarillo/rojo). 2) Top 5 alertas. 3) Plan de acción (3-5 pasos)."
        )
        res = llm.chat(provider=provider, model=model, system=system, user=user)
        alert_report = res.content if res.ok else ""

    if not alert_report:
        semaforo = "Verde"
        if atrasados or delta < -0.05:
            semaforo = "Amarillo"
        if delta < -0.12:
            semaforo = "Rojo"

        alert_report = (
            f"Semáforo: {semaforo}\n\n"
            "(Reporte generado sin LLM.)\n\n"
            f"Hitos atrasados: {', '.join(atrasados) if atrasados else 'N/A'}\n"
            f"Delta avance (actual-planeado): {delta:.2%}\n"
        )

    return ConstructionMonitorOutputs(milestones=milestones, summary=summary, alert_report=alert_report)


def lifecycle_status(baseline: pd.DataFrame, events: pd.DataFrame, as_of: date) -> dict:
    milestones, summary = compute_progress(baseline, events, as_of)
    planned = float(summary.loc[summary["metric"] == "planned_progress", "value"].iloc[0])
    actual = float(summary.loc[summary["metric"] == "actual_progress", "value"].iloc[0])
    delta = actual - planned
    atrasados = milestones[milestones["risk"] == "Atrasado"]["milestone"].tolist()

    future = milestones[~milestones["planned_completed"]].sort_values("planned_date")
    if not future.empty:
        next_m = future.iloc[0]
        next_milestone = str(next_m["milestone"])
        next_date = next_m["planned_date"]
        days_to_next = (next_date - as_of).days
    else:
        next_milestone = "Ninguno"
        next_date = None
        days_to_next = None

    if actual >= 1.0:
        phase = "Entrega y cierre"
    elif actual >= 0.75:
        phase = "Acabados e instalaciones finales"
    elif actual >= 0.40:
        phase = "Obra gris / estructura y mampostería"
    elif actual >= 0.10:
        phase = "Preliminares / cimentación"
    else:
        phase = "Inicio / movimientos de tierra"

    actions: list[str] = []
    if atrasados:
        actions.append(f"Recuperar hitos atrasados: {', '.join(atrasados)}.")
    if delta < -0.05:
        actions.append("Acelerar ritmo de obra para recuperar avance real vs planeado.")
    elif delta >= 0:
        actions.append("Mantener ritmo de obra; el avance está igual o adelantado.")
    if next_date:
        actions.append(f"Preparar el siguiente hito: {next_milestone} ({next_date.isoformat()}).")
    else:
        actions.append("Proyecto en cierre; coordinar entregas y puesta en marcha.")

    return {
        "phase": phase,
        "planned_progress": planned,
        "actual_progress": actual,
        "delta": delta,
        "atrasados": atrasados,
        "next_milestone": next_milestone,
        "next_date": next_date,
        "days_to_next": days_to_next,
        "actions": actions,
    }


def recommended_baseline(start_date: date, project_type: str = "residencial") -> pd.DataFrame:
    templates = {
        "residencial": [
            ("Cierro y replanteo", 15),
            ("Excavación", 30),
            ("Cimentación", 45),
            ("Estructura", 120),
            ("Mampostería", 75),
            ("Instalaciones", 90),
            ("Acabados", 105),
            ("Entrega", 15),
        ],
        "mixto": [
            ("Cierro y replanteo", 20),
            ("Excavación", 40),
            ("Cimentación", 55),
            ("Estructura", 150),
            ("Mampostería", 90),
            ("Instalaciones", 120),
            ("Acabados", 130),
            ("Entrega", 20),
        ],
    }
    tasks = templates.get(project_type, templates["residencial"])
    current = start_date
    rows: list[dict] = []
    for milestone, days in tasks:
        current += timedelta(days=days)
        rows.append({"milestone": milestone, "planned_date": current, "weight": days})
    return pd.DataFrame(rows)
