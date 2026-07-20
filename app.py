
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from src.agents.construction import run_construction_monitor
from src.agents.prefactibility import PrefactibilityInputs, run_prefactibility
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


def _build_chat_context() -> str:
    parts: list[str] = []

    pref_inputs = st.session_state.get("pref_inputs")
    pref_result = st.session_state.get("pref_result")
    if pref_inputs and pref_result:
        parts.append(
            f"Pre-factibilidad: ciudad={pref_inputs.city}, uso_suelo={pref_inputs.land_use}, "
            f"area_lote={pref_inputs.area_m2} m2, pisos_solicitados={pref_inputs.floors_requested}, "
            f"unidades={pref_inputs.units}, costo_lote=${pref_inputs.land_cost:,.0f}"
        )
        parts.append(
            f"Normativo: permitido={pref_result.normative.allowed}, "
            f"pisos_max={pref_result.normative.max_floors}, far_max={pref_result.normative.max_far}, "
            f"ocupacion_max={pref_result.normative.max_occupancy_ratio}"
        )
        irr = pref_result.finance.irr_annual
        irr_str = f"{irr:.1%}" if irr is not None else "N/D"
        parts.append(
            f"Finanzas: VAN={pref_result.finance.npv:,.0f}, "
            f"margen={pref_result.finance.profit_margin:.1%}, TIR_anual={irr_str}"
        )
        parts.append(f"Riesgos: {', '.join(pref_result.risks) if pref_result.risks else 'Ninguno'}")
        parts.append("Reporte ejecutivo de pre-factibilidad:\n" + pref_result.executive_report)
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
        atrasados = monitor_out.milestones[monitor_out.milestones["risk"] == "Atrasado"]["milestone"].tolist()
        parts.append(f"Hitos atrasados: {', '.join(atrasados) if atrasados else 'Ninguno'}")
        parts.append("Reporte de alertas:\n" + monitor_out.alert_report)
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


llm = MultiProviderLLM(secrets=st.secrets)

if "provider_name" not in st.session_state:
    st.session_state["provider_name"] = "OpenAI"
if "model" not in st.session_state:
    st.session_state["model"] = ""
if "use_llm" not in st.session_state:
    st.session_state["use_llm"] = True


def _llm_settings():
    provider = PROVIDERS[st.session_state.provider_name]
    model = st.session_state.get("model", "").strip() or None
    use_llm = st.session_state.get("use_llm", True)
    configured = llm.is_configured(provider)
    return provider, model, use_llm, configured


tab_pref, tab_monitor, tab_chat = st.tabs([
    "1) Pre-factibilidad",
    "2) Monitor de Obra",
    "3) Chat del Proyecto",
])


with tab_pref:
    st.subheader("Pre-factibilidad (Normativo/Financiero/Redactor)")
    provider, model, use_llm, configured = _llm_settings()
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
            provider=provider,
            model=model,
            use_llm=bool(use_llm and configured),
        )

        st.session_state["pref_inputs"] = inputs
        st.session_state["pref_result"] = result

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Normativo", "OK" if result.normative.allowed else "NO OK")
        k2.metric("VAN (NPV)", _money(result.finance.npv))
        k3.metric("Margen", f"{result.finance.profit_margin:.1%}")
        irr = result.finance.irr_annual
        k4.metric("TIR anual", f"{irr:.1%}" if irr is not None else "N/D")

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


with tab_monitor:
    st.subheader("Monitor de Ciclo de Vida de Obra (Extractor + Alertas)")
    provider, model, use_llm, configured = _llm_settings()

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

    if run_m:
        out = run_construction_monitor(
            baseline_df=baseline_df,
            events_df=events_df,
            as_of=as_of,
            llm=llm,
            provider=provider,
            model=model,
            use_llm=bool(use_llm and configured),
        )

        st.session_state["monitor_out"] = out

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


with tab_chat:
    st.subheader("Chat del Proyecto")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with st.expander("Configuración LLM", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.selectbox("Proveedor", list(PROVIDERS.keys()), key="provider_name")
        with col2:
            st.text_input("Modelo (opcional)", key="model")
        st.toggle("Usar LLM", key="use_llm")

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
            )
        st.session_state.messages = updated
        st.rerun()

