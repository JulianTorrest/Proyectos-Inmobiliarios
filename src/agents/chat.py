from __future__ import annotations

from typing import Optional

from src.llm.client import LLMResult, MultiProviderLLM
from src.llm.providers import LLMProviderConfig


def _system_prompt(context: str) -> str:
    return (
        "Eres un asesor senior en pre-factibilidad y seguimiento de obras inmobiliarias en Colombia. "
        "Responde en español, de forma clara y concisa, usando ÚNICAMENTE la información del contexto del proyecto. "
        "Si no hay datos suficientes para responder algo, di que no tienes esa información. "
        "NO inventes normativas oficiales ni cifras que no estén en el contexto. "
        "Recuerda que los datos son dummy de referencia, no oficiales.\n\n"
        f"--- CONTEXTO DEL PROYECTO ---\n{context}\n--- FIN CONTEXTO ---"
    )


def _fallback_answer(last_user_message: str, context: str) -> str:
    return (
        "No tengo acceso a un LLM en este momento (falta API key o el proveedor falló). "
        "Puedes activar un proveedor con llave en el sidebar o consultar directamente los datos en los tabs "
        "de Pre-factibilidad y Monitor de Obra."
    )


def chat_response(
    context: str,
    messages: list[dict[str, str]],
    llm: MultiProviderLLM,
    provider: LLMProviderConfig,
    model: Optional[str] = None,
    use_llm: bool = True,
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
        system=_system_prompt(context),
        messages=messages,
        temperature=0.2,
        max_tokens=1000,
    )

    if res.ok:
        answer = res.content
    else:
        answer = _fallback_answer(messages[-1]["content"], context)

    return answer, messages + [{"role": "assistant", "content": answer}]
