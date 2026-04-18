"""Prompts del agente.

Mantener estos textos aislados permite ajustarlos sin tocar la logica. Hablan
en espanol porque los tickets entrantes estan en espanol y el idioma del
prompt afecta la calidad de la respuesta.
"""

from __future__ import annotations

SYSTEM_PROMPT = """Eres el agente de soporte de una plataforma SaaS.
Tu tarea es leer un ticket recibido (ya clasificado por un modelo de ML) y decidir
una sola accion. Tu respuesta DEBE ser un objeto JSON valido, sin texto adicional,
con exactamente estas claves:

{
  "action": "auto_respond" | "escalate" | "request_info" | "close_duplicate",
  "reasoning": "explicacion breve de por que elegiste esta accion",
  "response_text": "mensaje al cliente en espanol si action == auto_respond o request_info, null en otro caso"
}

Criterios:
- auto_respond: el ticket es rutinario y puedes resolverlo con una respuesta directa
  (ej. recordatorio de como recuperar contrasena, donde encontrar la factura).
- escalate: urgencia alta o critica, sospecha de fraude, queja fuerte, dinero
  involucrado, daño reputacional, o duda donde se requiere intervencion humana.
- request_info: la informacion del ticket es insuficiente para decidir.
- close_duplicate: el ticket es practicamente identico a uno reciente del mismo usuario.

Responde SIEMPRE solo con el JSON. Sin prefijos, sin markdown, sin texto extra.
"""

USER_TEMPLATE = """TICKET
------
Usuario: {user_id}
Asunto: {subject}
Cuerpo: {body}

CLASIFICACION ML
----------------
Categoria: {category} (confianza {confidence_category:.2f})
Urgencia: {urgency} (confianza {confidence_urgency:.2f})

HISTORICO RECIENTE DEL USUARIO
------------------------------
{recent_history}

Decide la accion a tomar y responde SOLO con el JSON descrito.
"""


def render_user_prompt(
    *,
    user_id: str,
    subject: str,
    body: str,
    category: str,
    urgency: str,
    confidence_category: float,
    confidence_urgency: float,
    recent_history: str,
) -> str:
    return USER_TEMPLATE.format(
        user_id=user_id,
        subject=subject,
        body=body,
        category=category,
        urgency=urgency,
        confidence_category=confidence_category,
        confidence_urgency=confidence_urgency,
        recent_history=recent_history or "(Sin tickets previos)",
    )
