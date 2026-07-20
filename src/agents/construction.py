from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
