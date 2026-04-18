"""Generador de dataset sintetico reproducible.

El objetivo es tener un conjunto de entrenamiento "bueno pero no perfecto" que
permita ajustar un modelo con accuracy > 0.85 sin depender de datasets externos
(Kaggle cambia o exige cuenta). Cada categoria tiene ~20 plantillas y se
combinan con sufijos de urgencia para generar variacion controlada.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from src.db.models import TicketCategory, TicketUrgency

_SEED = 42


@dataclass(frozen=True)
class _Template:
    subject: str
    body: str
    category: TicketCategory
    urgency: TicketUrgency


_TEMPLATES: list[_Template] = [
    # --- Tecnico ---
    _Template("No carga la aplicacion", "Cuando abro la app me sale pantalla en blanco.", TicketCategory.TECNICO, TicketUrgency.ALTA),
    _Template("Error 500 al guardar", "Me aparece error 500 cuando guardo el formulario.", TicketCategory.TECNICO, TicketUrgency.ALTA),
    _Template("La pagina esta caida", "Todo el portal esta fuera de servicio para mi equipo.", TicketCategory.TECNICO, TicketUrgency.CRITICA),
    _Template("Se cierra solo el sistema", "El sistema se cierra solo despues de iniciar sesion.", TicketCategory.TECNICO, TicketUrgency.ALTA),
    _Template("Boton no responde", "El boton de enviar no hace nada en la pantalla de reportes.", TicketCategory.TECNICO, TicketUrgency.MEDIA),
    _Template("La app esta muy lenta", "Tarda mucho en cargar cada pantalla, antes era mas rapido.", TicketCategory.TECNICO, TicketUrgency.MEDIA),
    _Template("Fallo al subir archivo", "Me da error al intentar cargar un PDF de 2MB.", TicketCategory.TECNICO, TicketUrgency.MEDIA),
    _Template("Problema en el reporte PDF", "El PDF se descarga corrupto y no abre.", TicketCategory.TECNICO, TicketUrgency.MEDIA),
    _Template("Pantalla en blanco", "Al abrir el dashboard se queda todo en blanco.", TicketCategory.TECNICO, TicketUrgency.ALTA),
    _Template("Se pierde conexion", "La aplicacion pierde conexion cada dos minutos.", TicketCategory.TECNICO, TicketUrgency.ALTA),
    _Template("Error desconocido", "Me muestra un error desconocido al intentar hacer checkout.", TicketCategory.TECNICO, TicketUrgency.CRITICA),
    _Template("Actualizacion congelada", "Intento actualizar la app y se queda colgada al 50 por ciento.", TicketCategory.TECNICO, TicketUrgency.MEDIA),
    _Template("No guarda los cambios", "Edito mis datos y cuando recargo no se guardaron.", TicketCategory.TECNICO, TicketUrgency.MEDIA),
    _Template("Bug en el calendario", "Las fechas que selecciono se corren un dia.", TicketCategory.TECNICO, TicketUrgency.BAJA),
    _Template("Error ortografico en boton", "El boton dice cancelarr con dos erres.", TicketCategory.TECNICO, TicketUrgency.BAJA),
    _Template("No puedo imprimir", "Al darle imprimir no pasa nada.", TicketCategory.TECNICO, TicketUrgency.MEDIA),
    _Template("Fallo de integracion", "El webhook no esta llegando desde ayer.", TicketCategory.TECNICO, TicketUrgency.CRITICA),
    _Template("Se desconfigura el formato", "Los numeros aparecen sin separador de miles.", TicketCategory.TECNICO, TicketUrgency.BAJA),
    _Template("Boton desaparece en mobile", "En el celular el boton de pago no se ve.", TicketCategory.TECNICO, TicketUrgency.ALTA),
    _Template("No sincroniza datos", "Los cambios que hago en el movil no aparecen en el escritorio.", TicketCategory.TECNICO, TicketUrgency.MEDIA),

    # --- Facturacion ---
    _Template("Cobro incorrecto", "Me cobraron dos veces la mensualidad de este mes.", TicketCategory.FACTURACION, TicketUrgency.ALTA),
    _Template("No me llego la factura", "No he recibido la factura de marzo en mi correo.", TicketCategory.FACTURACION, TicketUrgency.MEDIA),
    _Template("Necesito cambiar datos de factura", "Debo corregir el NIT de facturacion en mi cuenta.", TicketCategory.FACTURACION, TicketUrgency.BAJA),
    _Template("Reembolso pendiente", "Llevo dos semanas esperando el reembolso del producto.", TicketCategory.FACTURACION, TicketUrgency.ALTA),
    _Template("Cobro despues de cancelar", "Cancele el servicio pero me siguieron cobrando.", TicketCategory.FACTURACION, TicketUrgency.CRITICA),
    _Template("Tarjeta declinada pero cobrada", "Me salio error al pagar pero el banco muestra el cobro.", TicketCategory.FACTURACION, TicketUrgency.CRITICA),
    _Template("Valor distinto al acordado", "Me cobraron 150 cuando el plan es de 120.", TicketCategory.FACTURACION, TicketUrgency.ALTA),
    _Template("No puedo pagar", "Mi tarjeta funciona en otros lados pero aqui la rechazan.", TicketCategory.FACTURACION, TicketUrgency.ALTA),
    _Template("Duda sobre mi factura", "No entiendo un cargo que aparece como servicio adicional.", TicketCategory.FACTURACION, TicketUrgency.BAJA),
    _Template("Solicitud de factura electronica", "Necesito la factura electronica para mi contador.", TicketCategory.FACTURACION, TicketUrgency.BAJA),
    _Template("Cambio de plan", "Quiero pasar del plan basico al plan pro.", TicketCategory.FACTURACION, TicketUrgency.BAJA),
    _Template("Actualizar tarjeta", "Necesito cambiar la tarjeta vinculada a mi cuenta.", TicketCategory.FACTURACION, TicketUrgency.MEDIA),
    _Template("Recibo erroneo", "El recibo trae un valor distinto al que pague.", TicketCategory.FACTURACION, TicketUrgency.MEDIA),
    _Template("Retraso en reembolso", "Ya pasaron 30 dias y el reembolso no llega.", TicketCategory.FACTURACION, TicketUrgency.ALTA),
    _Template("Cobro duplicado", "Aparece el mismo cargo dos veces en el extracto.", TicketCategory.FACTURACION, TicketUrgency.ALTA),
    _Template("Quiero renovar anualmente", "Me interesa pagar anual para tener descuento.", TicketCategory.FACTURACION, TicketUrgency.BAJA),
    _Template("Factura con datos incorrectos", "Salio con el nombre de otra empresa.", TicketCategory.FACTURACION, TicketUrgency.MEDIA),
    _Template("Pago rechazado sin motivo", "Me dice transaccion rechazada pero la tarjeta esta activa.", TicketCategory.FACTURACION, TicketUrgency.ALTA),
    _Template("Cobran mas de lo que consumo", "Solo usamos 3 usuarios y nos cobran por 10.", TicketCategory.FACTURACION, TicketUrgency.ALTA),
    _Template("Cambio de ciclo de facturacion", "Quiero mover mi facturacion al dia 15.", TicketCategory.FACTURACION, TicketUrgency.BAJA),

    # --- Cuenta ---
    _Template("No puedo iniciar sesion", "Puse mi correo y contrasena pero me dice credenciales invalidas.", TicketCategory.CUENTA, TicketUrgency.ALTA),
    _Template("Olvide mi contrasena", "Necesito recuperar el acceso a mi cuenta.", TicketCategory.CUENTA, TicketUrgency.MEDIA),
    _Template("Mi cuenta esta bloqueada", "Despues de varios intentos la cuenta quedo bloqueada.", TicketCategory.CUENTA, TicketUrgency.ALTA),
    _Template("No llega el correo de verificacion", "Me registre pero el correo de confirmacion no llega.", TicketCategory.CUENTA, TicketUrgency.MEDIA),
    _Template("Quiero cambiar mi correo", "Necesito actualizar el correo de mi cuenta.", TicketCategory.CUENTA, TicketUrgency.BAJA),
    _Template("Error al crear cuenta", "Al registrarme me sale error y no avanza.", TicketCategory.CUENTA, TicketUrgency.MEDIA),
    _Template("Usuarios duplicados", "Tengo dos cuentas con el mismo correo, quiero fusionarlas.", TicketCategory.CUENTA, TicketUrgency.BAJA),
    _Template("Acceso revocado", "Era administrador y ya no tengo permisos, pero nadie los cambio.", TicketCategory.CUENTA, TicketUrgency.CRITICA),
    _Template("No reconoce mi celular", "Tengo 2FA y ya no tengo el celular anterior, no puedo entrar.", TicketCategory.CUENTA, TicketUrgency.ALTA),
    _Template("Cuenta suspendida sin razon", "De la nada me dice cuenta suspendida, necesito entrar.", TicketCategory.CUENTA, TicketUrgency.CRITICA),
    _Template("Como elimino mi cuenta", "Quiero eliminar mi cuenta de forma definitiva.", TicketCategory.CUENTA, TicketUrgency.BAJA),
    _Template("Olvide mi usuario", "No recuerdo con que correo me registre.", TicketCategory.CUENTA, TicketUrgency.MEDIA),
    _Template("Error al actualizar datos", "Cuando cambio mi nombre no se guarda.", TicketCategory.CUENTA, TicketUrgency.BAJA),
    _Template("No puedo cerrar sesion", "El boton de cerrar sesion no responde.", TicketCategory.CUENTA, TicketUrgency.BAJA),
    _Template("Problema con 2FA", "El codigo que llega no funciona al ingresarlo.", TicketCategory.CUENTA, TicketUrgency.ALTA),
    _Template("Cuenta comprometida", "Creo que alguien accedio a mi cuenta sin autorizacion.", TicketCategory.CUENTA, TicketUrgency.CRITICA),
    _Template("Invitacion a miembro no llega", "Invite a un colega pero no le llega el correo.", TicketCategory.CUENTA, TicketUrgency.MEDIA),
    _Template("No puedo cambiar foto de perfil", "Subo la imagen y no se actualiza.", TicketCategory.CUENTA, TicketUrgency.BAJA),
    _Template("Sesion expira muy rapido", "Cada 5 minutos me pide iniciar sesion de nuevo.", TicketCategory.CUENTA, TicketUrgency.MEDIA),
    _Template("Agregar usuario al equipo", "Necesito sumar a mi asistente al plan.", TicketCategory.CUENTA, TicketUrgency.BAJA),

    # --- Informacion ---
    _Template("Como exporto mis datos", "Quiero descargar mis datos en CSV.", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Donde veo mis facturas", "No encuentro la seccion de facturacion en el panel.", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Cual es el horario de soporte", "Necesito saber en que horarios me pueden atender.", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Ofrecen plan empresarial", "Quiero saber si tienen un plan para empresas grandes.", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Integracion con Shopify", "Tienen integracion disponible con Shopify?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Como funciona el plan anual", "Me pueden explicar como funciona el descuento anual?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Tienen app movil", "Existe app para iOS y Android?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Documentacion del API", "Donde puedo leer la documentacion tecnica del API?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Webinar de induccion", "Hay algun webinar de induccion para equipos nuevos?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Politicas de privacidad", "Donde encuentro la politica de tratamiento de datos?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Versiones del software", "Que versiones soportan aun?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Pasos para migrar", "Cual es el proceso para migrar desde la competencia?", TicketCategory.INFORMACION, TicketUrgency.MEDIA),
    _Template("Cuantos usuarios trae el plan", "Cuantos usuarios incluye el plan pro?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Tienen soporte en espanol", "Brindan soporte en idioma espanol?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Ubicacion de servidores", "Donde estan alojados sus servidores?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Certificaciones de seguridad", "Tienen ISO 27001 o SOC 2?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("SLA del servicio", "Cual es el SLA que ofrecen?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Precios personalizados", "Manejan descuentos para ONGs?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Descarga de informes", "Como puedo descargar los reportes en Excel?", TicketCategory.INFORMACION, TicketUrgency.BAJA),
    _Template("Funcionalidades del plan", "Que incluye el plan basico exactamente?", TicketCategory.INFORMACION, TicketUrgency.BAJA),

    # --- Queja ---
    _Template("Mal servicio al cliente", "El asesor anterior fue muy grosero conmigo.", TicketCategory.QUEJA, TicketUrgency.MEDIA),
    _Template("Respuestas muy lentas", "Escribo por soporte y tardan dias en responder.", TicketCategory.QUEJA, TicketUrgency.MEDIA),
    _Template("Quiero cancelar el servicio", "Estoy muy insatisfecho y quiero darme de baja.", TicketCategory.QUEJA, TicketUrgency.ALTA),
    _Template("Promesa incumplida", "Me prometieron una funcion que nunca llego.", TicketCategory.QUEJA, TicketUrgency.MEDIA),
    _Template("Demasiadas caidas", "El servicio se cae cada semana, no es profesional.", TicketCategory.QUEJA, TicketUrgency.ALTA),
    _Template("Me cobraron sin permiso", "Autorice un cobro y me cargaron otros mas.", TicketCategory.QUEJA, TicketUrgency.CRITICA),
    _Template("Publicidad enganosa", "La landing decia algo distinto al producto real.", TicketCategory.QUEJA, TicketUrgency.MEDIA),
    _Template("Mal manejo de mis datos", "Me llegan correos de empresas que no conozco.", TicketCategory.QUEJA, TicketUrgency.ALTA),
    _Template("Calidad desmejorada", "Desde la ultima actualizacion todo funciona peor.", TicketCategory.QUEJA, TicketUrgency.MEDIA),
    _Template("Experiencia pesima", "Mi ultima experiencia fue muy mala, no recomiendo.", TicketCategory.QUEJA, TicketUrgency.MEDIA),
    _Template("Respuesta grosera", "Un asesor respondio de mala forma a mi solicitud.", TicketCategory.QUEJA, TicketUrgency.MEDIA),
    _Template("Espera de 3 horas", "Espere 3 horas en la linea sin respuesta.", TicketCategory.QUEJA, TicketUrgency.ALTA),
    _Template("Pierdo dinero por bugs", "Los errores del sistema me estan costando dinero.", TicketCategory.QUEJA, TicketUrgency.CRITICA),
    _Template("Nada funciona como dicen", "Prometen muchas cosas que no cumplen.", TicketCategory.QUEJA, TicketUrgency.MEDIA),
    _Template("Ya no lo recomiendo", "Llevo 6 meses decepcionado, no lo recomiendo.", TicketCategory.QUEJA, TicketUrgency.BAJA),
    _Template("Me cambian condiciones", "Me subieron el precio sin avisar.", TicketCategory.QUEJA, TicketUrgency.ALTA),
    _Template("Producto no es lo prometido", "Lo que recibi no corresponde a lo mostrado.", TicketCategory.QUEJA, TicketUrgency.MEDIA),
    _Template("Trato discriminatorio", "Me sentí discriminado por el trato recibido.", TicketCategory.QUEJA, TicketUrgency.ALTA),
    _Template("No se toman en serio mis reportes", "He reportado el mismo bug 5 veces y nada.", TicketCategory.QUEJA, TicketUrgency.MEDIA),
    _Template("Me obligan a renovar", "No me dejan cancelar y me insisten en renovar.", TicketCategory.QUEJA, TicketUrgency.ALTA),
]


_URGENCY_PREFIXES = {
    TicketUrgency.CRITICA: ["URGENTE: ", "CRITICO: ", "Emergencia: ", ""],
    TicketUrgency.ALTA: ["Por favor rapido, ", "Necesito ayuda, ", ""],
    TicketUrgency.MEDIA: ["", "", "Consulta: "],
    TicketUrgency.BAJA: ["Cuando puedan, ", "Sin prisa, ", ""],
}


def generate_dataset(samples_per_template: int = 5) -> pd.DataFrame:
    """Retorna un DataFrame con columnas subject, body, category, urgency."""
    rng = random.Random(_SEED)
    rows = []
    for tpl in _TEMPLATES:
        for _ in range(samples_per_template):
            prefix = rng.choice(_URGENCY_PREFIXES[tpl.urgency])
            subject = prefix + tpl.subject
            body = tpl.body
            if rng.random() < 0.3:
                body += " " + rng.choice(
                    [
                        "Agradezco su pronta respuesta.",
                        "Quedo atento.",
                        "Gracias de antemano.",
                        "Es importante.",
                        "Saludos cordiales.",
                    ]
                )
            rows.append(
                {
                    "subject": subject,
                    "body": body,
                    "category": tpl.category.value,
                    "urgency": tpl.urgency.value,
                }
            )
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def save_dataset(path: str, samples_per_template: int = 5) -> pd.DataFrame:
    df = generate_dataset(samples_per_template=samples_per_template)
    df.to_csv(path, index=False, encoding="utf-8")
    return df
