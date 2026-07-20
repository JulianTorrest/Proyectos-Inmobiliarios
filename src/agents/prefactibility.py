from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
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
            "NO inventes normativas reales: usa solo los datos proporcionados y aclara que son supuestos dummy. "
            "No uses emojis, iconos ni caracteres especiales como 📌, ⚠️, ✅, ❌. Usa solo texto plano y markdown simple."
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


def _metric_value(out: PrefactibilityOutputs, metric: str) -> float:
    if metric == "npv":
        return out.finance.npv
    if metric == "profit_margin":
        return out.finance.profit_margin
    if metric == "irr_annual":
        return out.finance.irr_annual if out.finance.irr_annual is not None else -float("inf")
    return out.finance.npv


def _quick_prefactibility(
    city: str,
    land_use: str,
    area_m2: float,
    floors_requested: int,
    units: int,
    avg_unit_size_m2: float,
    land_cost: float,
    rules_df: pd.DataFrame,
    market_df: pd.DataFrame,
    market_overrides: Optional[dict[str, float]] = None,
) -> PrefactibilityOutputs:
    rule = _pick_rule(rules_df, city, land_use)

    n_in = NormativeInputs(
        city=city,
        area_m2=area_m2,
        floors_requested=floors_requested,
        land_use=land_use,
    )
    n_out = evaluate_normative(n_in, rule)

    market = dict(_pick_market(market_df, city, land_use))
    if market_overrides:
        market.update(market_overrides)

    sell_price_per_unit = float(market["price_per_m2_sell"]) * float(avg_unit_size_m2)
    cost_per_m2_build = float(market["cost_per_m2_build"])
    soft_cost_pct = float(market["soft_cost_pct"])

    buildable_m2 = float(units) * float(avg_unit_size_m2)
    construction_cost_total = buildable_m2 * cost_per_m2_build
    soft_costs_total = construction_cost_total * soft_cost_pct

    f_in = FinanceInputs(
        total_units=int(units),
        sell_price_per_unit=float(sell_price_per_unit),
        construction_cost_total=float(construction_cost_total),
        soft_costs_total=float(soft_costs_total),
        land_cost=float(land_cost),
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

    return PrefactibilityOutputs(normative=n_out, finance=f_out, risks=risks, executive_report="")


def sensitivity_analysis(
    inputs: PrefactibilityInputs,
    rules_df: pd.DataFrame,
    market_df: pd.DataFrame,
    variables: tuple[str, ...] = (
        "units",
        "avg_unit_size_m2",
        "land_cost",
        "price_per_m2_sell",
        "cost_per_m2_build",
        "build_months",
        "sales_months",
    ),
    delta: float = 0.10,
) -> pd.DataFrame:
    base_market = _pick_market(market_df, inputs.city, inputs.land_use)
    base = _quick_prefactibility(
        inputs.city,
        inputs.land_use,
        inputs.area_m2,
        inputs.floors_requested,
        inputs.units,
        inputs.avg_unit_size_m2,
        inputs.land_cost,
        rules_df,
        market_df,
    ).finance
    rows: list[dict[str, Any]] = [
        {
            "variable": "Base",
            "direccion": "-",
            "npv": base.npv,
            "irr_annual": base.irr_annual,
            "profit_margin": base.profit_margin,
            "allowed": True,
        }
    ]
    for var in variables:
        for direction in (-1, 1):
            factor = 1.0 + direction * delta
            kwargs: dict[str, Any] = {}
            overrides: dict[str, float] = {}
            if var == "units":
                kwargs["units"] = max(1, int(inputs.units * factor))
            elif var == "avg_unit_size_m2":
                kwargs["avg_unit_size_m2"] = max(20.0, float(inputs.avg_unit_size_m2 * factor))
            elif var == "land_cost":
                kwargs["land_cost"] = float(inputs.land_cost * factor)
            elif var == "price_per_m2_sell":
                overrides["price_per_m2_sell"] = float(base_market["price_per_m2_sell"] * factor)
            elif var == "cost_per_m2_build":
                overrides["cost_per_m2_build"] = float(base_market["cost_per_m2_build"] * factor)
            elif var == "build_months":
                overrides["build_months"] = max(1, int(base_market["build_months"] * factor))
            elif var == "sales_months":
                overrides["sales_months"] = max(1, int(base_market["sales_months"] * factor))

            out = _quick_prefactibility(
                inputs.city,
                inputs.land_use,
                inputs.area_m2,
                inputs.floors_requested,
                kwargs.get("units", inputs.units),
                kwargs.get("avg_unit_size_m2", inputs.avg_unit_size_m2),
                kwargs.get("land_cost", inputs.land_cost),
                rules_df,
                market_df,
                market_overrides=overrides if overrides else None,
            )
            rows.append(
                {
                    "variable": var,
                    "direccion": f"{'-' if direction == -1 else '+'}{int(delta * 100)}%",
                    "npv": out.finance.npv,
                    "irr_annual": out.finance.irr_annual,
                    "profit_margin": out.finance.profit_margin,
                    "allowed": out.normative.allowed,
                }
            )
    return pd.DataFrame(rows)


def monte_carlo_prefactibility(
    inputs: PrefactibilityInputs,
    rules_df: pd.DataFrame,
    market_df: pd.DataFrame,
    n: int = 300,
    price_vol: float = 0.10,
    cost_vol: float = 0.10,
    units_vol: float = 0.05,
    land_cost_vol: float = 0.10,
    build_months_vol: float = 0.10,
    sales_months_vol: float = 0.10,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base_market = _pick_market(market_df, inputs.city, inputs.land_use)
    rows: list[dict[str, Any]] = []
    for _ in range(n):
        price_factor = 1.0 + rng.uniform(-price_vol, price_vol)
        cost_factor = 1.0 + rng.uniform(-cost_vol, cost_vol)
        units_factor = 1.0 + rng.uniform(-units_vol, units_vol)
        land_factor = 1.0 + rng.uniform(-land_cost_vol, land_cost_vol)
        bm_factor = 1.0 + rng.uniform(-build_months_vol, build_months_vol)
        sm_factor = 1.0 + rng.uniform(-sales_months_vol, sales_months_vol)
        new_units = max(1, int(inputs.units * units_factor))
        new_land_cost = float(inputs.land_cost * land_factor)
        overrides = {
            "price_per_m2_sell": float(base_market["price_per_m2_sell"] * price_factor),
            "cost_per_m2_build": float(base_market["cost_per_m2_build"] * cost_factor),
            "build_months": max(1, int(base_market["build_months"] * bm_factor)),
            "sales_months": max(1, int(base_market["sales_months"] * sm_factor)),
        }
        out = _quick_prefactibility(
            inputs.city,
            inputs.land_use,
            inputs.area_m2,
            inputs.floors_requested,
            new_units,
            inputs.avg_unit_size_m2,
            new_land_cost,
            rules_df,
            market_df,
            market_overrides=overrides,
        )
        rows.append(
            {
                "npv": out.finance.npv,
                "irr_annual": out.finance.irr_annual,
                "profit_margin": out.finance.profit_margin,
                "allowed": out.normative.allowed,
            }
        )
    return pd.DataFrame(rows)


def make_feasible(
    inputs: PrefactibilityInputs,
    rules_df: pd.DataFrame,
    market_df: pd.DataFrame,
    target_metric: str = "profit_margin",
    target_value: float = 0.15,
    iterations: int = 300,
) -> dict[str, Any]:
    base_market = _pick_market(market_df, inputs.city, inputs.land_use)
    base = _quick_prefactibility(
        inputs.city,
        inputs.land_use,
        inputs.area_m2,
        inputs.floors_requested,
        inputs.units,
        inputs.avg_unit_size_m2,
        inputs.land_cost,
        rules_df,
        market_df,
    )
    best_out = base
    best_params = {"price_mult": 1.0, "units_mult": 1.0, "land_mult": 1.0, "cost_mult": 1.0}
    best_score = _metric_value(base, target_metric)

    rng = np.random.default_rng(7)
    base_price = float(base_market["price_per_m2_sell"])
    base_cost = float(base_market["cost_per_m2_build"])

    for _ in range(iterations):
        price_mult = float(rng.uniform(0.85, 1.30))
        cost_mult = float(rng.uniform(0.80, 1.10))
        units_mult = float(rng.uniform(0.70, 1.30))
        land_mult = float(rng.uniform(0.70, 1.10))
        new_units = max(1, int(inputs.units * units_mult))
        new_land_cost = float(inputs.land_cost * land_mult)
        overrides = {
            "price_per_m2_sell": base_price * price_mult,
            "cost_per_m2_build": base_cost * cost_mult,
        }
        out = _quick_prefactibility(
            inputs.city,
            inputs.land_use,
            inputs.area_m2,
            inputs.floors_requested,
            new_units,
            inputs.avg_unit_size_m2,
            new_land_cost,
            rules_df,
            market_df,
            market_overrides=overrides,
        )
        val = _metric_value(out, target_metric)
        target_met = val >= target_value
        score = val + (1000.0 if target_met else 0.0)
        if score > best_score:
            best_score = score
            best_out = out
            best_params = {
                "price_mult": round(price_mult, 3),
                "units_mult": round(units_mult, 3),
                "land_mult": round(land_mult, 3),
                "cost_mult": round(cost_mult, 3),
                "new_units": new_units,
                "new_land_cost": new_land_cost,
            }

    return {
        "outputs": best_out,
        "params": best_params,
        "target_met": _metric_value(best_out, target_metric) >= target_value,
        "metric_value": _metric_value(best_out, target_metric),
    }


def recommend_unit_mix(
    inputs: PrefactibilityInputs,
    rules_df: pd.DataFrame,
    market_df: pd.DataFrame,
    min_units: int | None = None,
    max_units: int | None = None,
    step: int = 5,
) -> pd.DataFrame:
    if min_units is None:
        min_units = max(1, int(inputs.units * 0.6))
    if max_units is None:
        max_units = int(inputs.units * 1.5)
    rows: list[dict[str, Any]] = []
    for u in range(min_units, max_units + 1, step):
        out = _quick_prefactibility(
            inputs.city,
            inputs.land_use,
            inputs.area_m2,
            inputs.floors_requested,
            u,
            inputs.avg_unit_size_m2,
            inputs.land_cost,
            rules_df,
            market_df,
        )
        rows.append(
            {
                "units": u,
                "npv": out.finance.npv,
                "irr_annual": out.finance.irr_annual,
                "profit_margin": out.finance.profit_margin,
                "allowed": out.normative.allowed,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("profit_margin", ascending=False).reset_index(drop=True)


def design_advice(
    inputs: PrefactibilityInputs,
    rules_df: pd.DataFrame,
    llm: MultiProviderLLM,
    provider: LLMProviderConfig,
    model: Optional[str] = None,
    use_llm: bool = True,
) -> str:
    rule = _pick_rule(rules_df, inputs.city, inputs.land_use)
    if use_llm:
        system = (
            "Eres un arquitecto-asesor senior de pre-diseño inmobiliario en Colombia. "
            "Con base en los datos normativos y de mercado dummy entregados, da recomendaciones prácticas y concisas "
            "de diseño preliminar: torres, pisos, unidades por piso, parqueaderos, zonas comunes, y advertencias normativas. "
            "NO inventes normativas reales; aclara que los datos son supuestos. "
            "No uses emojis, iconos ni caracteres especiales como 📌, ⚠️, ✅, ❌. Usa solo texto plano y markdown simple."
        )
        user = (
            f"Ciudad: {inputs.city}, uso de suelo: {inputs.land_use}, área lote: {inputs.area_m2} m2, "
            f"pisos solicitados: {inputs.floors_requested}, unidades: {inputs.units}, "
            f"tamaño promedio: {inputs.avg_unit_size_m2} m2, costo lote: ${inputs.land_cost:,.0f} COP.\n\n"
            f"Regla normativa dummy: max_floors={rule.max_floors}, max_far={rule.max_far}, "
            f"max_occupancy_ratio={rule.max_occupancy_ratio}, max_height_m={rule.max_height_m}.\n\n"
            "Entrega bullets de diseño preliminar y advertencias."
        )
        res = llm.chat(provider=provider, model=model, system=system, user=user)
        if res.ok:
            return res.content
    max_buildable_far = inputs.area_m2 * rule.max_far
    units_far_max = int(max_buildable_far / inputs.avg_unit_size_m2) if inputs.avg_unit_size_m2 else 0
    floors = min(inputs.floors_requested, rule.max_floors)
    units_per_floor = max(1, int(inputs.units / max(inputs.floors_requested, 1)))
    towers = max(1, int(np.ceil(units_far_max / max(floors * units_per_floor, 1))))
    parking = int(inputs.units / 1.5)
    lines = [
        f"Altura máxima permitida: {rule.max_floors} pisos / {rule.max_height_m} m.",
        f"FAR máximo: {rule.max_far}, equivalente a ~{max_buildable_far:,.0f} m2 construidos.",
        f"Con {inputs.avg_unit_size_m2} m2/unidad, el techo por FAR es ~{units_far_max} unidades.",
        f"Configuración sugerida: {towers} torre(s) de ~{floors} pisos con ~{units_per_floor} unidades/piso.",
        f"Parqueaderos estimados: ~{parking} cupos (aprox. 1 por cada 1.5 unidades).",
    ]
    if inputs.floors_requested > rule.max_floors:
        lines.append(
            f"AVISO: los {inputs.floors_requested} pisos solicitados exceden el máximo permitido ({rule.max_floors}); "
            "reducir o tramitar excepción."
        )
    return "\n- ".join([""] + lines)


def generate_checklist(
    inputs: PrefactibilityInputs,
    rules_df: pd.DataFrame,
    llm: MultiProviderLLM,
    provider: LLMProviderConfig,
    model: Optional[str] = None,
    use_llm: bool = True,
) -> str:
    rule = _pick_rule(rules_df, inputs.city, inputs.land_use)
    if use_llm:
        system = (
            "Eres un experto en trámites inmobiliarios en Colombia. "
            "Elabora un checklist de pasos legales, permisos y viabilidades necesarios para desarrollar un proyecto "
            "con los datos entregados. Sé pragmático, indica responsable sugerido y riesgo de demora. "
            "NO inventes normativas oficiales; aclara que es orientativo y requiere validación legal. "
            "No uses emojis, iconos ni caracteres especiales como 📌, ⚠️, ✅, ❌. Usa solo texto plano y markdown simple."
        )
        user = (
            f"Ciudad: {inputs.city}\n"
            f"Uso de suelo: {inputs.land_use}\n"
            f"Área del lote: {inputs.area_m2} m2\n"
            f"Pisos solicitados: {inputs.floors_requested}\n"
            f"Unidades: {inputs.units}\n"
            f"Altura máxima permitida: {rule.max_floors} pisos / {rule.max_height_m} m\n\n"
            "Entrega un checklist numerado de 8-12 pasos con: paso, responsable sugerido, riesgo de demora (alto/medio/bajo)."
        )
        res = llm.chat(provider=provider, model=model, system=system, user=user)
        if res.ok:
            return res.content
    return (
        "Checklist de viabilidad (orientativo, sin LLM):\n"
        "1. Verificación de usos de suelo y normativa aplicable.\n"
        "2. Estudios previos de riesgo y usos del suelo.\n"
        "3. Concepto de planeación urbana / licencia preliminar.\n"
        "4. Licencia de urbanización o construcción según corresponda.\n"
        "5. Aprobación de planos arquitectónicos y estructurales.\n"
        "6. Permisos de uso de acueducto, energía y gas.\n"
        "7. Gestión ambiental y manejo de residuos.\n"
        "8. Pólizas de cumplimiento y contratos de obra.\n"
    )
