from __future__ import annotations

from typing import Optional

from src.llm.client import LLMResult, MultiProviderLLM
from src.llm.providers import LLMProviderConfig


ROLE_PROMPTS = {
    "CEO": "Actúa como un CEO/Consejero: piensa en visión estratégica, riesgos de gobierno, veredicto de inversión y resumen ejecutivo. Sé directo.",
    "CFO": "Actúa como un CFO: enfócate en VAN, TIR, margen, estructura de costos, sensibilidad, punto de equilibrio y riesgos financieros. Sé preciso con cifras.",
    "COO": "Actúa como un COO: enfócate en cronograma, hitos, productividad, cuellos de botella, mitigaciones y eficiencia operativa.",
    "CMO": "Actúa como un CMO: enfócate en posicionamiento, mix de unidades, precios, segmento objetivo, diferenciación y comercialización.",
    "General": "Actúa como un asesor inmobiliario senior: equilibra normativa, finanzas, operación y mercado.",
}


FEW_SHOT = """
Ejemplo 1:
Pregunta: ¿Cuál es el VAN del proyecto?
Respuesta: El VAN calculado es $X millones COP, basado en los ingresos por venta de Y unidades y los costos estimados (construcción, soft costs y lote). Considera que los datos son dummy.

Ejemplo 2:
Pregunta: ¿Es viable normativamente construir 15 pisos?
Respuesta: Normativamente el máximo permitido es Z pisos para ese uso de suelo. Solicitar 15 pisos excedería el límite, por lo que no sería viable sin revisión. Consulta la planeación oficial antes de decisiones reales.

Ejemplo 3:
Pregunta: ¿Cómo va la obra?
Respuesta: Al corte, el avance real es A% vs un planeado de B%, con un delta de C puntos porcentuales. Los hitos atrasados son: [...]. Se recomienda acelerar [...].
"""


def _system_prompt(context: str, role: str, tone: str, cot: bool, language: str) -> str:
    role_prompt = ROLE_PROMPTS.get(role, ROLE_PROMPTS["General"])
    tone_prompt = (
        "Responde como para un ejecutivo: en 3-5 bullets, directo, sin tecnicismos innecesarios, con recomendación de acción clara."
        if tone == "Ejecutivo"
        else "Responde como analista: detallado, estructurado, muestra supuestos, pasos y matices. Usa tablas o listas cuando ayude."
    )
    cot_prompt = (
        "\nAntes de dar la respuesta final, piensa paso a paso brevemente (máximo 3 líneas de razonamiento) y luego entrega la conclusión.\n"
        if cot
        else ""
    )
    lang_prompt = "Responde en español." if language.lower().startswith("es") else "Responde en inglés."

    return (
        "Eres un asesor senior en pre-factibilidad y seguimiento de obras inmobiliarias en Colombia. "
        f"{role_prompt}\n"
        f"{tone_prompt}\n"
        "Usa ÚNICAMENTE la información del contexto del proyecto. "
        "Si no hay datos suficientes para responder algo, di que no tienes esa información. "
        "NO inventes normativas oficiales ni cifras que no estén en el contexto. "
        "Recuerda que los datos son dummy de referencia, no oficiales.\n"
        f"{lang_prompt}\n"
        f"{cot_prompt}\n"
        "Aquí tienes ejemplos de formato esperado:\n"
        f"{FEW_SHOT}\n\n"
        f"--- CONTEXTO DEL PROYECTO ---\n{context}\n--- FIN CONTEXTO ---"
    )


def _fallback_answer(last_user_message: str, context: str) -> str:
    return (
        "No tengo acceso a un LLM en este momento (falta API key o el proveedor falló). "
        "Revisa la API key en Streamlit Cloud > Secrets o variables de entorno, "
        "y el proveedor/modelo en la Configuración LLM del chat."
    )


def chat_response(
    context: str,
    messages: list[dict[str, str]],
    llm: MultiProviderLLM,
    provider: LLMProviderConfig,
    model: Optional[str] = None,
    use_llm: bool = True,
    role: str = "General",
    tone: str = "Ejecutivo",
    cot: bool = False,
    language: str = "Español",
) -> tuple[str, list[dict[str, str]]]:
    """Return assistant answer and the updated conversation history.

    `messages` must contain the current user message already.
    """
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("messages must end with a user message")

    if not use_llm:
        answer = _fallback_answer(messages[-1]["content"], context)
        return answer, messages + [{"role": "assistant", "content": answer}]

    res = llm.chat(
        provider=provider,
        model=model,
        system=_system_prompt(context, role, tone, cot, language),
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
    )

    if res.ok:
        answer = res.content
    else:
        answer = (
            "No pude contactar al LLM. Error del proveedor:\n\n"
            f"```\n{res.error}\n```\n\n"
            "Revisa la API key, el nombre del modelo y la conexión a internet."
        )

    return answer, messages + [{"role": "assistant", "content": answer}]
