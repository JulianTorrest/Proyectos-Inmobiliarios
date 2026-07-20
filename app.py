
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import io

import pandas as pd
import plotly.express as px
import streamlit as st
from fpdf import FPDF

from src.agents.construction import run_construction_monitor
from src.domain.construction_monitor import compute_progress
from src.agents.prefactibility import (
    PrefactibilityInputs,
    design_advice,
    generate_checklist,
    make_feasible,
    monte_carlo_prefactibility,
    recommend_unit_mix,
    run_prefactibility,
    sensitivity_analysis,
)
from src.data.loaders import (
    load_baseline_schedule,
    load_market_assumptions,
    load_normative_rules,
    load_projects_large,
    load_site_events,
    load_site_events_large,
)
from src.agents import chat as chat_agent
from src.llm.client import MultiProviderLLM
from src.llm.providers import PROVIDERS


def _money(v: float) -> str:
    return f"${v:,.0f}"


def _pdf_safe(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def _generate_project_pdf() -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, _pdf_safe("Reporte Ejecutivo - Proyecto Inmobiliario"), ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, _pdf_safe("1. Resumen del proyecto"), ln=True)
    pdf.set_font("Arial", "", 10)
    description = (
        "Prototipo multiagente para evaluacion de pre-factibilidad y monitoreo de obras inmobiliarias en Colombia. "
        "Los datos son dummy de referencia, no oficiales."
    )
    pdf.multi_cell(0, 5, _pdf_safe(description))
    pdf.ln(3)

    pref_inputs = st.session_state.get("pref_inputs")
    pref_result = st.session_state.get("pref_result")
    if pref_inputs and pref_result:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, _pdf_safe("2. Pre-factibilidad"), ln=True)
        pdf.set_font("Arial", "", 10)
        irr = pref_result.finance.irr_annual
        irr_str = f"{irr:.1%}" if irr is not None else "N/D"
        lines = [
            f"Ciudad: {pref_inputs.city}",
            f"Uso de suelo: {pref_inputs.land_use}",
            f"Area del lote: {pref_inputs.area_m2:,.0f} m2",
            f"Pisos solicitados: {pref_inputs.floors_requested}",
            f"Unidades: {pref_inputs.units}",
            f"Tamano promedio: {pref_inputs.avg_unit_size_m2} m2",
            f"Costo del lote: {_money(pref_inputs.land_cost)}",
            f"VAN: {_money(pref_result.finance.npv)}",
            f"TIR: {irr_str}",
            f"Margen: {pref_result.finance.profit_margin:.1%}",
            f"Permitido: {'Si' if pref_result.normative.allowed else 'No'}",
        ]
        for line in lines:
            pdf.cell(0, 5, _pdf_safe(line), ln=True)
        pdf.ln(3)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, _pdf_safe("Reporte ejecutivo"), ln=True)
        pdf.set_font("Arial", "", 10)
        for paragraph in pref_result.executive_report.split("\n"):
            pdf.multi_cell(0, 5, _pdf_safe(paragraph))
            pdf.ln(1)
    else:
        pdf.cell(0, 5, _pdf_safe("No se ha ejecutado la pre-factibilidad."), ln=True)

    monitor_out = st.session_state.get("monitor_out")
    if monitor_out:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, _pdf_safe("3. Monitor de obra"), ln=True)
        pdf.set_font("Arial", "", 10)
        summary = monitor_out.summary.set_index("metric")
        planned = float(summary.loc["planned_progress", "value"])
        actual = float(summary.loc["actual_progress", "value"])
        delta = float(summary.loc["delta", "value"])
        pdf.cell(0, 5, _pdf_safe(f"Avance planeado: {planned:.1%}"), ln=True)
        pdf.cell(0, 5, _pdf_safe(f"Avance real: {actual:.1%}"), ln=True)
        pdf.cell(0, 5, _pdf_safe(f"Delta: {delta:.1%}"), ln=True)
        atrasados = monitor_out.milestones[monitor_out.milestones["risk"] == "Atrasado"]["milestone"].tolist()
        pdf.cell(0, 5, _pdf_safe(f"Hitos atrasados: {', '.join(atrasados) if atrasados else 'Ninguno'}"), ln=True)
        pdf.ln(3)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 6, _pdf_safe("Alertas"), ln=True)
        pdf.set_font("Arial", "", 10)
        for paragraph in monitor_out.alert_report.split("\n"):
            pdf.multi_cell(0, 5, _pdf_safe(paragraph))
            pdf.ln(1)
    else:
        pdf.cell(0, 5, _pdf_safe("No se ha ejecutado el monitor de obra."), ln=True)

    return bytes(pdf.output(dest="S"))


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


def _build_chat_context() -> str:
    parts: list[str] = [
        "Este es un prototipo de aplicación multiagente para evaluación de pre-factibilidad "
        "y monitoreo de obras inmobiliarias en Colombia. Combina agentes de normativa, "
        "finanzas, redacción, extracción de datos de obra y alertas. "
        "Los datos son dummy de referencia, no oficiales."
    ]

    rules_df = load_normative_rules()
    market_df = load_market_assumptions()
    baseline_df = load_baseline_schedule()
    events_df = load_site_events()

    pref_inputs = st.session_state.get("pref_inputs")
    if pref_inputs:
        rule_row = rules_df[
            (rules_df["city"] == pref_inputs.city) & (rules_df["land_use"] == pref_inputs.land_use)
        ].head(1)
        if not rule_row.empty:
            r = rule_row.iloc[0]
            parts.append(
                f"Normativa aplicable (RAG) para {pref_inputs.city}/{pref_inputs.land_use}: "
                f"max_floors={r['max_floors']}, max_far={r['max_far']}, "
                f"max_occupancy={r['max_occupancy_ratio']}, max_height_m={r['max_height_m']}, "
                f"notas={r.get('notes', '')}"
            )
        market_row = market_df[
            (market_df["city"] == pref_inputs.city) & (market_df["land_use"] == pref_inputs.land_use)
        ].head(1)
        if not market_row.empty:
            m = market_row.iloc[0]
            parts.append(
                f"Supuestos de mercado (RAG) para {pref_inputs.city}/{pref_inputs.land_use}: "
                f"precio_m2_venta={m['price_per_m2_sell']}, costo_m2_construccion={m['cost_per_m2_build']}, "
                f"soft_cost_pct={m['soft_cost_pct']}, meses_construccion={m['build_months']}, "
                f"meses_venta={m['sales_months']}, tasa_descuento_anual={m['discount_rate_annual']}"
            )
    else:
        parts.append("Normativa disponible (RAG):\n" + rules_df.head(20).to_csv(index=False))
        parts.append("Mercado disponible (RAG):\n" + market_df.head(20).to_csv(index=False))

    parts.append("Cronograma baseline (RAG):\n" + baseline_df.head(20).to_csv(index=False))
    recent_events = events_df.sort_values("event_date", ascending=False).head(20)
    parts.append("Eventos recientes (RAG):\n" + recent_events.to_csv(index=False))

    pref_result = st.session_state.get("pref_result")
    if pref_inputs and pref_result:
        parts.append(
            f"Entradas pre-factibilidad (completas): ciudad={pref_inputs.city}, uso_suelo={pref_inputs.land_use}, "
            f"area_lote={pref_inputs.area_m2} m2, pisos_solicitados={pref_inputs.floors_requested}, "
            f"unidades={pref_inputs.units}, tamano_promedio_unidad={pref_inputs.avg_unit_size_m2} m2, "
            f"costo_lote=${pref_inputs.land_cost:,.0f}"
        )
        parts.append(
            f"Resultado normativo completo: permitido={pref_result.normative.allowed}, "
            f"pisos_max={pref_result.normative.max_floors}, far_max={pref_result.normative.max_far}, "
            f"ocupacion_max={pref_result.normative.max_occupancy_ratio}, "
            f"altura_max={pref_result.normative.max_height_m}, "
            f"razones={' / '.join(pref_result.normative.reasons) if pref_result.normative.reasons else 'Ninguna'}"
        )
        irr = pref_result.finance.irr_annual
        irr_str = f"{irr:.1%}" if irr is not None else "N/D"
        parts.append(
            f"Métricas financieras completas: VAN={pref_result.finance.npv:,.0f}, "
            f"margen={pref_result.finance.profit_margin:.1%}, TIR_anual={irr_str}, "
            f"ingresos={pref_result.finance.revenue_total:,.0f}, "
            f"costos={pref_result.finance.costs_total:,.0f}, "
            f"utilidad={pref_result.finance.profit_total:,.0f}"
        )
        parts.append(f"Riesgos identificados: {', '.join(pref_result.risks) if pref_result.risks else 'Ninguno'}")
        parts.append("Reporte ejecutivo de pre-factibilidad:\n" + pref_result.executive_report)
        checklist = st.session_state.get("pref_checklist")
        if checklist:
            parts.append("Checklist de viabilidad generado:\n" + checklist)
        design = st.session_state.get("design_advice_text")
        if design:
            parts.append("Asesor de diseño preliminar:\n" + design)
    else:
        parts.append("No se ha ejecutado aún la pre-factibilidad.")

    monitor_out = st.session_state.get("monitor_out")
    if monitor_out:
        summary = monitor_out.summary.set_index("metric")
        planned = float(summary.loc["planned_progress", "value"])
        actual = float(summary.loc["actual_progress", "value"])
        delta = float(summary.loc["delta", "value"])
        parts.append(
            f"Monitor de obra: avance_planeado={planned:.1%}, avance_real={actual:.1%}, delta={delta:.1%}"
        )
        parts.append("Resumen del monitor de obra (CSV):\n" + monitor_out.summary.to_csv(index=False))
        parts.append("Hitos del monitor de obra (CSV):\n" + monitor_out.milestones.to_csv(index=False))
        atrasados = monitor_out.milestones[monitor_out.milestones["risk"] == "Atrasado"]["milestone"].tolist()
        parts.append(f"Hitos atrasados: {', '.join(atrasados) if atrasados else 'Ninguno'}")
        parts.append("Reporte de alertas del monitor de obra:\n" + monitor_out.alert_report)
    else:
        parts.append("No se ha ejecutado aún el monitor de obra.")

    return "\n\n".join(parts)


def _read_csv_upload(upload) -> Optional[pd.DataFrame]:
    if upload is None:
        return None
    try:
        return pd.read_csv(upload)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _cached_projects_large(n: int, seed: int) -> pd.DataFrame:
    return load_projects_large(n=n, seed=seed)


@st.cache_data(show_spinner=False)
def _cached_site_events_large(n: int, seed: int) -> pd.DataFrame:
    return load_site_events_large(n=n, seed=seed)


st.set_page_config(page_title="Multiagentes Inmobiliario (Colombia)", layout="wide")

st.title("Evaluador Automatizado y Monitor de Obra (Multiagente)")

pdf_bytes = _generate_project_pdf()
st.download_button(
    "Descargar PDF ejecutivo",
    data=pdf_bytes,
    file_name="reporte_proyecto.pdf",
    mime="application/pdf",
)


llm = MultiProviderLLM(secrets=st.secrets)

if "provider_name" not in st.session_state:
    st.session_state["provider_name"] = "OpenAI"
if "model" not in st.session_state:
    st.session_state["model"] = ""
if "use_llm" not in st.session_state:
    st.session_state["use_llm"] = True
if "chat_role" not in st.session_state:
    st.session_state["chat_role"] = "General"
if "chat_tone" not in st.session_state:
    st.session_state["chat_tone"] = "Ejecutivo"
if "chat_cot" not in st.session_state:
    st.session_state["chat_cot"] = False
if "chat_language" not in st.session_state:
    st.session_state["chat_language"] = "Español"


def _llm_settings():
    provider = PROVIDERS[st.session_state.provider_name]
    model = st.session_state.get("model", "").strip() or None
    use_llm = st.session_state.get("use_llm", True)
    configured = llm.is_configured(provider)
    return provider, model, use_llm, configured


tab_pref, tab_monitor, tab_chat, tab_info, tab_arch = st.tabs([
    "1) Pre-factibilidad",
    "2) Monitor de Obra",
    "3) Chat del Proyecto",
    "4) Info del Proyecto",
    "5) Arquitectura",
])


with tab_pref:
    st.subheader("Pre-factibilidad (Normativo/Financiero/Redactor)")
    provider, model, use_llm, configured = _llm_settings()
    mistral_provider = PROVIDERS["Mistral"]
    mistral_configured = llm.is_configured(mistral_provider)
    rules_df = load_normative_rules()
    market_df = load_market_assumptions()

    with st.expander("Datos dummy (volumen)"):
        use_big_projects = st.toggle("Usar dataset grande de proyectos (>=100k)", value=False)
        big_n_projects = st.number_input("N proyectos", min_value=100_000, value=100_000, step=50_000)
        big_seed_projects = st.number_input("Seed proyectos", min_value=0, value=7, step=1)

        if use_big_projects:
            st.caption("Se genera por código y se cachea (no se guarda en git).")
            projects_df = _cached_projects_large(int(big_n_projects), int(big_seed_projects))
            st.dataframe(projects_df.head(50), use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        city_options = sorted(market_df["city"].unique())
        city = st.selectbox("Ciudad", city_options, index=0)
        land_use_options = sorted(market_df[market_df["city"] == city]["land_use"].unique())
        land_use = st.selectbox("Uso de suelo (dummy)", land_use_options, index=0)
    with col_b:
        area_m2 = st.number_input("Área del lote (m2)", min_value=200.0, value=1200.0, step=50.0)
        floors_requested = st.number_input("Pisos solicitados", min_value=1, value=12, step=1)
    with col_c:
        units = st.number_input("# Unidades", min_value=1, value=120, step=5)
        avg_unit_size_m2 = st.number_input("Tamaño promedio unidad (m2)", min_value=20.0, value=65.0, step=1.0)
        land_cost = st.number_input("Costo del lote (COP)", min_value=0.0, value=12_000_000_000.0, step=100_000_000.0)

    run = st.button("Evaluar pre-factibilidad", type="primary")

    if run or "pref_result" in st.session_state:
        if run:
            inputs = PrefactibilityInputs(
                city=city,
                land_use=land_use,
                area_m2=float(area_m2),
                floors_requested=int(floors_requested),
                units=int(units),
                avg_unit_size_m2=float(avg_unit_size_m2),
                land_cost=float(land_cost),
            )
            result = run_prefactibility(
                inputs=inputs,
                normative_rules_df=rules_df,
                market_df=market_df,
                llm=llm,
                provider=mistral_provider,
                model=None,
                use_llm=mistral_configured,
            )

            st.session_state["pref_inputs"] = inputs
            st.session_state["pref_result"] = result
        else:
            inputs = st.session_state["pref_inputs"]
            result = st.session_state["pref_result"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Normativo", "OK" if result.normative.allowed else "NO OK")
        k2.metric("VAN (NPV)", _money(result.finance.npv))
        k3.metric("Margen", f"{result.finance.profit_margin:.1%}")
        irr = result.finance.irr_annual
        k4.metric("TIR anual", f"{irr:.1%}" if irr is not None else "N/D")

        norm_msg = (
            "Normativo OK: la configuración cumple con los límites dummy de pisos, FAR, ocupación y altura."
            if result.normative.allowed
            else "Normativo NO OK: la configuración NO cumple uno o más límites normativos dummy. Revisa:"
        )
        st.caption(norm_msg)
        if not result.normative.allowed:
            for r in result.normative.reasons:
                st.write(f"- {r}")

        with st.expander("Glosario de métricas"):
            st.markdown(
                """
- **VAN (NPV)**: Valor Actual Neto. Suma de los flujos de caja futuros descontados a hoy. Si es **positivo**, el proyecto genera valor; si es **negativo**, no cubre la tasa de descuento.
- **TIR anual**: Tasa Interna de Retorno. Es la rentabilidad anualizada del proyecto. Si la TIR es mayor a la tasa de descuento, el proyecto es financieramente atractivo.
- **Margen (profit margin)**: Utilidad neta dividida por ingresos totales. Indica qué porcentaje de cada peso vendido se convierte en ganancia.
- **Normativo OK/NO OK**: Indica si la configuración de pisos, área y unidades cumple con las reglas dummy de normativa del caso.
"""
            )

        st.markdown("### Resumen financiero")
        fin_df = pd.DataFrame(
            {
                "concepto": ["Ingresos", "Costos", "Utilidad"],
                "valor": [
                    result.finance.revenue_total,
                    result.finance.costs_total,
                    result.finance.profit_total,
                ],
            }
        )
        st.plotly_chart(
            px.bar(fin_df, x="concepto", y="valor", text_auto=True, title="Ingresos vs Costos"),
            use_container_width=True,
        )

        st.markdown("### Riesgos detectados")
        if result.risks:
            for r in result.risks:
                st.write(f"- {r}")
        else:
            st.write("- N/A")

        st.markdown("### Reporte ejecutivo")
        st.text_area("", value=result.executive_report, height=260)

        st.markdown("---")
        st.subheader("Optimización y análisis avanzado")

        if "pref_result" in st.session_state:
            inputs = st.session_state["pref_inputs"]
            result = st.session_state["pref_result"]

            st.markdown("### Asesor de diseño preliminar")
            st.text_area(
                "",
                value=design_advice(
                    inputs,
                    rules_df,
                    llm,
                    mistral_provider,
                    model=None,
                    use_llm=mistral_configured,
                ),
                height=140,
                key="design_advice_text",
            )

            with st.expander("Guardar y comparar escenarios", expanded=False):
                c1, c2 = st.columns([1, 3])
                with c1:
                    if st.button("Guardar escenario actual"):
                        scenarios = st.session_state.get("pref_scenarios", [])
                        scenarios.append(
                            {
                                "nombre": f"Esc {len(scenarios) + 1}",
                                "unidades": int(inputs.units),
                                "pisos": int(inputs.floors_requested),
                                "costo_lote": float(inputs.land_cost),
                                "VAN": float(result.finance.npv),
                                "margen": float(result.finance.profit_margin),
                                "TIR": result.finance.irr_annual,
                            }
                        )
                        st.session_state["pref_scenarios"] = scenarios
                        st.rerun()
                with c2:
                    if st.session_state.get("pref_scenarios"):
                        st.dataframe(pd.DataFrame(st.session_state["pref_scenarios"]), use_container_width=True)

            st.markdown("### Análisis de sensibilidad (±10%)")
            sens_df = sensitivity_analysis(inputs, rules_df, market_df)
            st.dataframe(sens_df, use_container_width=True)
            st.plotly_chart(
                px.bar(sens_df, x="variable", y="npv", color="direccion", barmode="group", title="Impacto en VAN"),
                use_container_width=True,
            )

            st.markdown("### Simulación Monte Carlo")
            n_sim = st.number_input(
                "Número de simulaciones",
                min_value=50,
                max_value=2000,
                value=300,
                step=50,
                key="mc_n",
            )
            if st.button("Correr Monte Carlo"):
                with st.spinner("Simulando..."):
                    mc_df = monte_carlo_prefactibility(inputs, rules_df, market_df, n=int(n_sim))
                st.session_state["mc_df"] = mc_df
            if "mc_df" in st.session_state:
                mc_df = st.session_state["mc_df"]
                st.dataframe(mc_df.describe(), use_container_width=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(px.histogram(mc_df, x="npv", nbins=30, title="Distribución VAN"), use_container_width=True)
                with c2:
                    st.plotly_chart(px.histogram(mc_df, x="profit_margin", nbins=30, title="Distribución Margen"), use_container_width=True)

            st.markdown("### Hazlo factible")
            target_metric = st.selectbox(
                "Métrica objetivo",
                ["profit_margin", "npv", "irr_annual"],
                key="target_metric",
            )
            if target_metric == "profit_margin":
                target_value = st.number_input("Margen objetivo (%)", value=15.0, step=1.0, key="target_value") / 100.0
            elif target_metric == "npv":
                target_value = st.number_input(
                    "VAN objetivo (COP)",
                    value=0.0,
                    step=1_000_000_000.0,
                    key="target_value",
                )
            else:
                target_value = st.number_input("TIR objetivo (%)", value=15.0, step=1.0, key="target_value") / 100.0
            if st.button("Optimizar"):
                with st.spinner("Buscando ajustes..."):
                    feasible = make_feasible(inputs, rules_df, market_df, target_metric=target_metric, target_value=target_value)
                st.session_state["feasible_result"] = feasible
            if "feasible_result" in st.session_state:
                feasible = st.session_state["feasible_result"]
                st.write("**Mejor escenario encontrado:**")
                st.json(feasible["params"])
                if target_metric == "npv":
                    st.metric("VAN alcanzado", _money(feasible["metric_value"]))
                else:
                    st.metric("Métrica alcanzada", f"{feasible['metric_value']:.2%}")
                st.write(f"Meta alcanzada: {'Sí' if feasible['target_met'] else 'No'}")

            st.markdown("### Recomendación de mix de unidades")
            step_mix = st.number_input(
                "Paso de unidades",
                min_value=1,
                max_value=50,
                value=5,
                step=1,
                key="mix_step",
            )
            mix_df = recommend_unit_mix(inputs, rules_df, market_df, step=int(step_mix))
            st.dataframe(mix_df.head(10), use_container_width=True)
            st.plotly_chart(
                px.line(mix_df, x="units", y="profit_margin", title="Margen por número de unidades"),
                use_container_width=True,
            )

            st.markdown("### Comparables de mercado")
            st.dataframe(market_df, use_container_width=True)

            st.markdown("### Exportar resultados")
            export_rows = [
                ["Ciudad", inputs.city],
                ["Uso de suelo", inputs.land_use],
                ["Área lote (m2)", inputs.area_m2],
                ["Pisos solicitados", inputs.floors_requested],
                ["Unidades", inputs.units],
                ["Tamaño promedio (m2)", inputs.avg_unit_size_m2],
                ["Costo lote", inputs.land_cost],
                ["VAN", result.finance.npv],
                [
                    "TIR",
                    result.finance.irr_annual if result.finance.irr_annual is not None else "N/D",
                ],
                ["Margen", result.finance.profit_margin],
                ["Ingresos", result.finance.revenue_total],
                ["Costos", result.finance.costs_total],
                ["Utilidad", result.finance.profit_total],
                ["Permitido", result.normative.allowed],
            ]
            export_df = pd.DataFrame(export_rows, columns=["concepto", "valor"])
            col_csv, col_xlsx = st.columns(2)
            with col_csv:
                st.download_button(
                    "Descargar CSV",
                    data=export_df.to_csv(index=False).encode("utf-8"),
                    file_name="prefactibilidad.csv",
                    mime="text/csv",
                )
            with col_xlsx:
                buffer = io.BytesIO()
                export_df.to_excel(buffer, index=False, sheet_name="Prefactibilidad")
                st.download_button(
                    "Descargar Excel",
                    data=buffer.getvalue(),
                    file_name="prefactibilidad.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            st.markdown("### Checklist de viabilidad")
            if st.button("Generar checklist con LLM", key="btn_checklist"):
                with st.spinner("Generando checklist..."):
                    checklist = generate_checklist(
                        inputs,
                        rules_df,
                        llm,
                        mistral_provider,
                        model=None,
                        use_llm=mistral_configured,
                    )
                st.session_state["pref_checklist"] = checklist
            if "pref_checklist" in st.session_state:
                st.text_area("", value=st.session_state["pref_checklist"], height=240, key="checklist_text")


with tab_monitor:
    st.subheader("Monitor de Ciclo de Vida de Obra (Extractor + Alertas)")
    provider, model, use_llm, configured = _llm_settings()
    mistral_provider = PROVIDERS["Mistral"]
    mistral_configured = llm.is_configured(mistral_provider)

    left, right = st.columns([1, 2])
    with left:
        as_of = st.date_input("Fecha de corte", value=date(2026, 9, 30))
        st.caption("El cronograma y los eventos se cargan automaticamente desde data/.")
        use_big_events = st.toggle("Generar eventos alternativos (>=100k)", value=False)
        if use_big_events:
            big_n_events = st.number_input("N eventos", min_value=100_000, value=100_000, step=50_000)
            big_seed_events = st.number_input("Seed eventos", min_value=0, value=11, step=1)
        run_m = st.button("Evaluar avance vs baseline", type="primary")

    baseline_df = load_baseline_schedule()
    if use_big_events:
        events_df = _cached_site_events_large(int(big_n_events), int(big_seed_events))
    else:
        events_df = load_site_events()

    if run_m or "monitor_out" in st.session_state:
        if run_m:
            out = run_construction_monitor(
                baseline_df=baseline_df,
                events_df=events_df,
                as_of=as_of,
                llm=llm,
                provider=mistral_provider,
                model=None,
                use_llm=mistral_configured,
            )

            st.session_state["monitor_out"] = out
        else:
            out = st.session_state["monitor_out"]

        with right:
            st.markdown("### Resumen")
            summary = out.summary.set_index("metric")
            st.metric("Avance planeado", f"{float(summary.loc['planned_progress','value']):.1%}")
            st.metric("Avance real", f"{float(summary.loc['actual_progress','value']):.1%}")
            st.metric("Delta", f"{float(summary.loc['delta','value']):.1%}")

            st.markdown("### Hitos")
            st.dataframe(
                out.milestones[["milestone", "planned_date", "completed", "actual_date", "risk", "delay_days"]],
                use_container_width=True,
            )

            chart_df = out.milestones.copy()
            chart_df["planned_date"] = pd.to_datetime(chart_df["planned_date"])
            chart_df["actual_date"] = pd.to_datetime(chart_df["actual_date"], errors="coerce")

            st.plotly_chart(
                px.scatter(
                    chart_df,
                    x="planned_date",
                    y="milestone",
                    color="risk",
                    title="Hitos (fecha planeada) y riesgo",
                ),
                use_container_width=True,
            )

            st.markdown("### Reporte de alertas")
            st.text_area("", value=out.alert_report, height=260)

            st.markdown("---")
            st.subheader("Ciclo de vida y cronograma")

            life = lifecycle_status(baseline_df, events_df, as_of)
            st.metric("Fase actual", life["phase"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Avance planeado", f"{life['planned_progress']:.1%}")
            c2.metric("Avance real", f"{life['actual_progress']:.1%}")
            c3.metric(
                "Días al próximo hito",
                life["days_to_next"] if life["days_to_next"] is not None else "N/A",
            )
            if life["next_date"]:
                st.caption(f"Próximo hito: {life['next_milestone']} ({life['next_date']})")
            st.markdown("**Acciones recomendadas:**")
            for a in life["actions"]:
                st.write(f"- {a}")

            st.markdown("### Cronograma recomendado")
            project_type = st.selectbox("Tipo de proyecto", ["residencial", "mixto"], key="project_type")
            start_b = st.date_input("Fecha de inicio", value=date(2026, 1, 1), key="rec_start")
            if st.button("Generar baseline sugerido", key="btn_rec_baseline"):
                rec_df = recommended_baseline(start_b, project_type=project_type)
                st.session_state["rec_baseline"] = rec_df
            if "rec_baseline" in st.session_state:
                st.dataframe(st.session_state["rec_baseline"], use_container_width=True)
                csv = st.session_state["rec_baseline"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Descargar baseline CSV",
                    data=csv,
                    file_name="baseline_recomendado.csv",
                    mime="text/csv",
                )

            st.markdown("### Exportar monitor")
            mon_csv = out.milestones.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar hitos CSV", data=mon_csv, file_name="monitor_hitos.csv", mime="text/csv")


with tab_chat:
    st.subheader("Chat del Proyecto")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.expander("Configuración LLM y Estilo", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Proveedor", list(PROVIDERS.keys()), key="provider_name")
        with col2:
            st.text_input("Modelo (opcional)", key="model")
        st.toggle("Usar LLM", key="use_llm")

        st.markdown("---")
        role_col, tone_col, lang_col = st.columns(3)
        with role_col:
            st.selectbox("Rol C-level", ["General", "CEO", "CFO", "COO", "CMO"], key="chat_role")
        with tone_col:
            st.selectbox("Tono", ["Ejecutivo", "Analista"], key="chat_tone")
        with lang_col:
            st.selectbox("Idioma", ["Español", "English"], key="chat_language")
        st.checkbox("Razonar paso a paso (Chain-of-Thought)", key="chat_cot")

    with st.expander("Historial del chat", expanded=False):
        if st.session_state.messages:
            chat_df = pd.DataFrame(st.session_state.messages)
            csv = chat_df.to_csv(index=False).encode("utf-8")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Descargar historial CSV",
                    data=csv,
                    file_name="chat_historial.csv",
                    mime="text/csv",
                )
            with c2:
                if st.button("Borrar historial"):
                    st.session_state.messages = []
                    st.rerun()
        else:
            st.caption("Aún no hay mensajes.")

    provider, model, use_llm, configured = _llm_settings()
    if use_llm and not configured:
        st.warning(f"Falta API key: {provider.api_key_env}. Se usará modo heurístico.")
    if use_llm and configured:
        st.success(f"LLM listo ({provider.name}).")

    if user_prompt := st.chat_input("Haz una pregunta sobre el proyecto..."):
        new_messages = list(st.session_state.messages) + [{"role": "user", "content": user_prompt}]
        ctx = _build_chat_context()
        with st.spinner("Pensando..."):
            answer, updated = chat_agent.chat_response(
                context=ctx,
                messages=new_messages,
                llm=llm,
                provider=provider,
                model=model,
                use_llm=bool(use_llm and configured),
                role=st.session_state.get("chat_role", "General"),
                tone=st.session_state.get("chat_tone", "Ejecutivo"),
                cot=st.session_state.get("chat_cot", False),
                language=st.session_state.get("chat_language", "Español"),
            )
        st.session_state.messages = updated
        st.rerun()


with tab_info:
    st.subheader("Información del Proyecto")
    st.markdown(
        """
## 1. ¿Qué es esto?
Prototipo de una aplicación multiagente para evaluar **pre-factibilidad** y monitorear el **ciclo de vida de obras inmobiliarias** en Colombia. El objetivo es acelerar decisiones C-level con evaluaciones normativas, financieras y de riesgo usando LLMs.

## 2. ¿Cómo lo hicimos?
- **Streamlit** como UI y motor de reruns/estado (`st.session_state`).
- **Agentes** en `src/agents/`: `prefactibility.py` (evaluación + reporte + diseño + checklist), `construction.py` (monitor + alertas), `chat.py` (asistente conversacional).
- **Lógica de dominio** en `src/domain/` para mantener los cálculos puros y testeables.
- **Datos** en `data/` CSV dummy, cargados/cachéados con `src/data/loaders.py` y generados con `src/data/generate.py` y scripts en `scripts/`.
- **LLM multi-proveedor** en `src/llm/`: un wrapper sobre el SDK de OpenAI que puede apuntar a Mistral, OpenAI u otros endpoints compatibles.

## 2.1. ¿Usamos LangChain?
No. Este prototipo **no usa LangChain**. Usamos un wrapper propio (`MultiProviderLLM` en `src/llm/client.py`) sobre el SDK oficial de `openai` para llamar a Mistral y otros endpoints compatibles. Se hizo así para:
- Mantener el stack ligero y evitar dependencias adicionales en un MVP.
- Tener control directo de prompts, contexto RAG y parámetros de generación.
- Facilitar la migración futura a LangChain, LlamaIndex u otro framework si el proyecto crece.

LangChain sería útil si en el futuro queremos orquestar agentes con herramientas, memoria a largo plazo o RAG vectorial con embeddings.

## 3. ¿Por qué solo funciona Mistral?
En `src/llm/providers.py` se definen los proveedores y sus variables de entorno (`MISTRAL_API_KEY`, `OPENAI_API_KEY`, etc.). La app usa `resolve_api_key` para buscar la API key en `secrets.toml` o en variables de entorno. Hoy **solo `MISTRAL_API_KEY` está configurada**, por eso forzamos Mistral para reporte ejecutivo, checklist, asesor de diseño y alertas. Si se agregan las demás keys, `MultiProviderLLM` las usará automáticamente.

## 4. Frameworks y librerías
- **Streamlit**: prototipado rápido de UI sin front-end separado.
- **pandas / numpy / scipy**: manipulación y cálculos numéricos.
- **plotly**: visualizaciones interactivas.
- **openai SDK**: llamadas a modelos compatibles (Mistral usa el mismo formato).
- **fpdf2 + openpyxl**: exportación de reportes PDF y Excel.

## 5. Funcionalidades actuales
- Evaluación normativa (pisos, FAR, ocupación, altura) y financiera (VAN, TIR, margen).
- Reporte ejecutivo con LLM.
- Asesor de diseño preliminar con LLM.
- Checklist de viabilidad con LLM.
- Monte Carlo, optimización "hazlo factible" y recomendación de mix de unidades.
- Comparación de escenarios.
- Monitor de obra vs baseline con alertas LLM.
- Cronograma recomendado y estado de ciclo de vida.
- Export CSV/Excel/PDF.
- Chat del proyecto con contexto RAG de pre-factibilidad y monitor.

## 6. Funcionalidades intentadas / no implementadas aún
- Integración real con bases de normativa/planeación urbana (usa datos dummy).
- Carga de documentos reales (PDF, DWG, planos); solo soporta CSV dummy.
- Persistencia en base de datos (todo vive en `session_state` y CSV locales).
- Autenticación y multiusuario.
- RAG vectorial con embeddings (se usa contexto plano en texto; no hay vector DB).
- Uso de OpenAI u otros proveedores por defecto (requieren sus API keys).

## 7. Archivos principales y relaciones
- `app.py`: orquesta toda la UI y el flujo.
- `src/agents/prefactibility.py`: agente de pre-factibilidad.
- `src/agents/construction.py`: agente de monitor de obra.
- `src/agents/chat.py`: agente de chat.
- `src/domain/normative.py`, `finance.py`, `construction_monitor.py`: cálculos de dominio.
- `src/data/loaders.py`: carga de CSV; `src/data/generate.py`: generación de datos dummy grandes.
- `src/llm/client.py`: wrapper LLM; `src/llm/providers.py`: configuraciones.
- `scripts/generate_dummy.py` y `populate_data.py`: scripts para poblar datos.
- `data/*.csv`: archivos dummy de normativa, mercado, baseline y eventos.
- `requirements.txt`: dependencias.
"""
    )


with tab_arch:
    st.subheader("Arquitectura")
    st.markdown(
        """
A continuación hay dos diagramas:
1. **Relación de archivos del repositorio**: qué archivo importa/usa a cuál.
2. **Arquitectura macro**: cómo fluyen datos y decisiones entre el usuario, Streamlit, agentes, dominio, datos y LLM.
"""
    )

    st.subheader("Relación de archivos")

    file_dot = r"""
digraph Archivos {
    rankdir=TB;
    node [shape=box, style="rounded,filled", fillcolor="#E8F4F8", fontname="Helvetica"];
    edge [fontname="Helvetica"];

    app [label="app.py\nStreamlit UI / orquestador"];
    agents [label="src/agents/*.py\nprefactibility | construction | chat"];
    domain [label="src/domain/*.py\nnormative | finance | construction_monitor"];
    data [label="src/data/*.py\nloaders | generate"];
    llm [label="src/llm/*.py\nclient | providers"];
    scripts [label="scripts/*.py\ngenerate_dummy | populate_data"];
    csv [label="data/*.csv\nnormativa | mercado | baseline | eventos", shape=cylinder];
    apis [label="Mistral / OpenAI APIs", shape=ellipse, fillcolor="#FFF8E1"];

    app -> agents [label="coordina"];
    app -> data [label="carga CSV"];
    agents -> domain [label="calcula"];
    agents -> llm [label="chat"];
    llm -> apis [label="OpenAI SDK"];
    data -> csv [label="lee"];
    scripts -> csv [label="genera"];
}
"""

    st.graphviz_chart(file_dot, use_container_width=True)

    st.subheader("Arquitectura macro")

    macro_dot = r"""
digraph Macro {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fillcolor="#F3E8FF", fontname="Helvetica"];
    edge [fontname="Helvetica"];

    usuario [label="Usuario C-level", shape=ellipse];
    streamlit [label="Streamlit UI\n(app.py)"];
    agents [label="Agentes LLM\nprefact | monitor | chat"];
    dominio [label="Lógica de dominio\npandas / numpy"];
    datos [label="Datos dummy\nCSV + loaders"];
    llm [label="Mistral / OpenAI\nOpenAI SDK", shape=ellipse, fillcolor="#FFF8E1"];
    reportes [label="Reportes\nPDF / Excel / CSV"];

    usuario -> streamlit -> agents -> dominio -> datos;
    agents -> llm;
    streamlit -> reportes;
    dominio -> reportes;
}
"""

    st.graphviz_chart(macro_dot, use_container_width=True)
