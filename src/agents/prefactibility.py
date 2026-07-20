from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from src.domain.finance import FinanceInputs, FinanceOutputs, evaluate_project
from src.domain.normative import NormativeInputs, NormativeOutputs, NormativeRule, evaluate_normative
from src.llm.client import MultiProviderLLM
from src.llm.providers import LLMProviderConfig


@dataclass(frozen=True)
class PrefactibilityInputs:
    city: str
    land_use: str
    area_m2: float
    floors_requested: int
    units: int
    avg_unit_size_m2: float
    land_cost: float


@dataclass(frozen=True)
class PrefactibilityOutputs:
    normative: NormativeOutputs
    finance: FinanceOutputs
    risks: list[str]
    executive_report: str


def _pick_rule(rules_df: pd.DataFrame, city: str, land_use: str) -> NormativeRule:
    row = (
        rules_df[(rules_df["city"] == city) & (rules_df["land_use"] == land_use)]
        .head(1)
        .to_dict(orient="records")
    )
    if not row:
        raise ValueError("No se encontró regla normativa dummy para la ciudad/uso seleccionado.")

    r = row[0]
    return NormativeRule(
        city=str(r["city"]),
        land_use=str(r["land_use"]),
        max_floors=int(r["max_floors"]),
        max_height_m=float(r["max_height_m"]),
        max_occupancy_ratio=float(r["max_occupancy_ratio"]),
        max_far=float(r["max_far"]),
        notes=str(r.get("notes", "")),
    )


def _pick_market(market_df: pd.DataFrame, city: str, land_use: str) -> dict[str, Any]:
    row = (
        market_df[(market_df["city"] == city) & (market_df["land_use"] == land_use)]
        .head(1)
        .to_dict(orient="records")
    )
    if not row:
        raise ValueError("No se encontró supuesto de mercado dummy para la ciudad/uso seleccionado.")
    return row[0]


def run_prefactibility(
    inputs: PrefactibilityInputs,
    normative_rules_df: pd.DataFrame,
    market_df: pd.DataFrame,
    llm: MultiProviderLLM,
    provider: LLMProviderConfig,
    model: Optional[str] = None,
    use_llm: bool = True,
) -> PrefactibilityOutputs:
    rule = _pick_rule(normative_rules_df, inputs.city, inputs.land_use)

    n_in = NormativeInputs(
        city=inputs.city,
        area_m2=inputs.area_m2,
        floors_requested=inputs.floors_requested,
        land_use=inputs.land_use,
    )
    n_out = evaluate_normative(n_in, rule)

    market = _pick_market(market_df, inputs.city, inputs.land_use)

    sell_price_per_unit = float(market["price_per_m2_sell"]) * float(inputs.avg_unit_size_m2)
    cost_per_m2_build = float(market["cost_per_m2_build"])
    soft_cost_pct = float(market["soft_cost_pct"])

    buildable_m2 = float(inputs.units) * float(inputs.avg_unit_size_m2)
    construction_cost_total = buildable_m2 * cost_per_m2_build
    soft_costs_total = construction_cost_total * soft_cost_pct

    f_in = FinanceInputs(
        total_units=int(inputs.units),
        sell_price_per_unit=float(sell_price_per_unit),
        construction_cost_total=float(construction_cost_total),
        soft_costs_total=float(soft_costs_total),
        land_cost=float(inputs.land_cost),
        sales_months=int(market["sales_months"]),
        build_months=int(market["build_months"]),
        discount_rate_annual=float(market["discount_rate_annual"]),
    )
    f_out = evaluate_project(f_in)

    risks: list[str] = []
    if not n_out.allowed:
        risks.extend(n_out.reasons)

    if f_out.profit_margin < 0.15:
        risks.append("Margen esperado bajo (< 15%).")

    if f_out.npv < 0:
        risks.append("VAN negativo con la tasa de descuento seleccionada.")

    executive_report = ""

    if use_llm:
        system = (
            "Eres un consultor senior de pre-factibilidad inmobiliaria en Colombia. "
            "Debes redactar un reporte ejecutivo breve, con decisiones y riesgos. "
            "NO inventes normativas reales: usa solo los datos proporcionados y aclara que son supuestos dummy."
        )
        user = (
            f"Ciudad: {inputs.city}\n"
            f"Uso de suelo: {inputs.land_use}\n"
            f"Área del lote (m2): {inputs.area_m2}\n"
            f"Pisos solicitados: {inputs.floors_requested}\n"
            f"Unidades: {inputs.units}\n"
            f"Tamaño promedio unidad (m2): {inputs.avg_unit_size_m2}\n"
            f"Costo del lote: {inputs.land_cost}\n\n"
            f"Resultado normativo dummy: allowed={n_out.allowed}, max_floors={n_out.max_floors}, max_far={n_out.max_far}, "
            f"max_occupancy_ratio={n_out.max_occupancy_ratio}.\n"
            f"Supuestos mercado dummy: price_per_m2_sell={market['price_per_m2_sell']}, cost_per_m2_build={market['cost_per_m2_build']}, "
            f"soft_cost_pct={market['soft_cost_pct']}, build_months={market['build_months']}, sales_months={market['sales_months']}.\n\n"
            f"Finanzas: revenue_total={f_out.revenue_total:.0f}, costs_total={f_out.costs_total:.0f}, profit_total={f_out.profit_total:.0f}, "
            f"profit_margin={f_out.profit_margin:.2%}, NPV={f_out.npv:.0f}, IRR_annual={f_out.irr_annual}.\n\n"
            f"Riesgos detectados: {', '.join(risks) if risks else 'N/A'}\n\n"
            "Entrega: 1) Veredicto (viable/no viable/condicional). 2) 5 bullets de insights. 3) Matriz de riesgos (alto/medio/bajo)."
        )
        res = llm.chat(provider=provider, model=model, system=system, user=user)
        executive_report = res.content if res.ok else ""

    if not executive_report:
        verdict = "Viable" if (n_out.allowed and f_out.npv > 0) else "Condicional"
        if (not n_out.allowed) or (f_out.npv < 0):
            verdict = "No viable"

        executive_report = (
            f"Veredicto: {verdict}\n\n"
            "(Reporte generado sin LLM: usando reglas y supuestos dummy.)\n\n"
            f"Riesgos: {', '.join(risks) if risks else 'N/A'}\n"
        )

    return PrefactibilityOutputs(
        normative=n_out,
        finance=f_out,
        risks=risks,
        executive_report=executive_report,
    )
