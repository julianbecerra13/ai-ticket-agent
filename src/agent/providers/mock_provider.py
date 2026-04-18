"""Proveedor mock: sirve para tests y para el modo demo sin API key.

Genera respuestas plausibles basadas en la categoria/urgencia enviadas en el
prompt del usuario, sin llamar a ningun servicio externo.
"""

from __future__ import annotations

import json
import re

from src.agent.providers.base import LLMProvider
from src.db.models import AgentActionType, TicketCategory, TicketUrgency

_AUTO_RESPONSES = {
    TicketCategory.CUENTA: (
        "Hola, para recuperar el acceso usa la opcion 'Olvide mi contrasena' en la pantalla "
        "de inicio de sesion. Recibiras un correo con el enlace de recuperacion en pocos minutos."
    ),
    TicketCategory.FACTURACION: (
        "Hola, puedes consultar y descargar tus facturas en la seccion 'Facturacion' dentro "
        "del menu de configuracion de tu cuenta."
    ),
    TicketCategory.INFORMACION: (
        "Gracias por escribir. Toda la informacion sobre planes, integraciones y documentacion "
        "esta disponible en nuestra pagina de ayuda. Si necesitas algo mas especifico, con gusto te guiamos."
    ),
    TicketCategory.TECNICO: (
        "Lamentamos el inconveniente. Intenta cerrar sesion, limpiar la cache del navegador y "
        "volver a entrar. Si el error persiste, respondenos con una captura de pantalla."
    ),
    TicketCategory.QUEJA: (
        "Lamentamos tu experiencia. Hemos registrado tu caso y un asesor humano se pondra en "
        "contacto contigo en menos de 24 horas."
    ),
}


class MockProvider(LLMProvider):
    name = "mock"
    model = "mock-v1"

    def generate(self, *, system: str, user: str) -> str:
        category = _extract(user, r"Categoria:\s*(\w+)", default=TicketCategory.INFORMACION.value)
        urgency = _extract(user, r"Urgencia:\s*(\w+)", default=TicketUrgency.MEDIA.value)

        try:
            cat = TicketCategory(category)
        except ValueError:
            cat = TicketCategory.INFORMACION
        try:
            urg = TicketUrgency(urgency)
        except ValueError:
            urg = TicketUrgency.MEDIA

        if urg in (TicketUrgency.ALTA, TicketUrgency.CRITICA) or cat == TicketCategory.QUEJA:
            action = AgentActionType.ESCALATE
            response_text = None
            reasoning = (
                f"Urgencia {urg.value} y categoria {cat.value}: se requiere revision humana."
            )
        elif cat == TicketCategory.TECNICO and urg == TicketUrgency.MEDIA:
            action = AgentActionType.REQUEST_INFO
            response_text = (
                "Podrias indicarnos navegador, sistema operativo y un video corto del error?"
            )
            reasoning = "Se necesita mas contexto tecnico antes de resolver."
        else:
            action = AgentActionType.AUTO_RESPOND
            response_text = _AUTO_RESPONSES.get(cat, _AUTO_RESPONSES[TicketCategory.INFORMACION])
            reasoning = f"Caso rutinario de {cat.value}; se responde con la plantilla estandar."

        return json.dumps(
            {
                "action": action.value,
                "reasoning": reasoning,
                "response_text": response_text,
            },
            ensure_ascii=False,
        )


def _extract(text: str, pattern: str, *, default: str) -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default
