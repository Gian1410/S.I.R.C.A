"""Agentes estructurados y gobernados para expedientes preventivos de Caja Trujillo.

No es un chatbot: cada salida se valida con Pydantic, está limitada al JSON de
políticas y cuenta con un modo de respaldo determinista para demostraciones.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

try:
    from dotenv import load_dotenv
except ImportError:  # Permite usar el respaldo incluso si dotenv no fue instalado.
    load_dotenv = None


RUTA_BASE = Path(__file__).resolve().parent
MODELO_GEMINI = "gemini-2.5-flash"


class DiagnosticoRiesgo(BaseModel):
    diagnostico_tecnico: str = Field(min_length=1, max_length=1200)
    instrumento_financiero_sugerido: str = Field(min_length=1, max_length=300)
    monto_o_condicion_aplicable: str = Field(min_length=1, max_length=700)
    justificacion_normativa: str = Field(min_length=1, max_length=1200)
    requiere_visita_campo: bool


class MedidaOperativa(BaseModel):
    accion: str = Field(min_length=1, max_length=500)
    costo_estimado_pen: float = Field(ge=0)
    plazo_ejecucion_dias: int = Field(ge=0)


class PlanContinuidad(BaseModel):
    plan_accion_operativo: list[MedidaOperativa] = Field(min_length=2, max_length=3)
    mensaje_whatsapp_cliente: str = Field(min_length=1, max_length=450)


def cargar_politicas(ruta: str | Path | None = None) -> dict[str, Any]:
    """Carga la fuente de verdad de gobernanza de las etapas previas."""
    archivo = Path(ruta) if ruta else RUTA_BASE / "politicas_y_reglas.json"
    with archivo.open("r", encoding="utf-8") as manejador:
        return json.load(manejador)


def _amenaza_maxima(cliente: dict[str, Any]) -> float:
    return max(
        float(cliente.get("amenaza_inundacion_escenario", cliente.get("amenaza_inundacion_base", 0.0))),
        float(cliente.get("amenaza_sequia_escenario", cliente.get("amenaza_sequia_base", 0.0))),
    )


def _hechos_suficientes_para_instrumento(cliente: dict[str, Any]) -> tuple[bool, list[str]]:
    """Verifica hechos mínimos; datos ausentes nunca se presumen favorables."""
    faltantes: list[str] = []
    if _amenaza_maxima(cliente) < 0.55:
        faltantes.append("amenaza base/escenario menor a 0.55")
    if int(float(cliente.get("dias_atraso_actual", 999))) > 15:
        faltantes.append("atraso mayor a 15 días o no informado")
    if not bool(cliente.get("plan_adaptacion_verificado", False)):
        faltantes.append("plan de adaptación y destino trazable aún no verificados")
    if not bool(cliente.get("capacidad_pago_evaluada", False)):
        faltantes.append("evaluación de capacidad de pago aún no verificada")
    if str(cliente.get("nivel_confianza_alerta", "Baja")) == "Baja":
        faltantes.append("nivel de confianza de alerta Baja")
    return len(faltantes) == 0, faltantes


def _respaldo_riesgo(cliente: dict[str, Any], politicas: dict[str, Any]) -> DiagnosticoRiesgo:
    """Produce una recomendación repetible, fundada y conservadora sin red/API."""
    cumple, observaciones = _hechos_suficientes_para_instrumento(cliente)
    riesgo_fisico = _amenaza_maxima(cliente)
    prob = float(cliente.get("prob_estres_liquidez", 0.0))
    holgura = float(cliente.get("holgura_financiera", 0.0))
    fragilidad = "fragilidad financiera" if holgura < 1.0 or prob >= 0.50 else "capacidad financiera relativamente resistente"
    diagnostico = (
        f"Amenaza física estimada: {riesgo_fisico:.2f}; probabilidad de estrés de liquidez: {prob:.0%}. "
        f"El riesgo combina exposición climática y {fragilidad}; no constituye una causal de exclusión crediticia."
    )
    if cumple:
        micro = politicas["politicas_riesgo_caja_trujillo"]["microcredito_preventivo_adaptacion"]
        instrumento = "Microcrédito Preventivo de Adaptación sujeto a aprobación humana"
        condicion = (
            f"Hasta S/ {micro['monto_maximo_pen']:,.0f}, respetando el límite adicional y el destino trazable; "
            "confirmar presupuesto y no sobreendeudamiento."
        )
        norma = "Microcrédito Preventivo de Adaptación: se verificaron las condiciones disponibles y se requiere aprobación humana final."
    else:
        instrumento = "Sin instrumento financiero automático; activar gestión preventiva y validación humana"
        condicion = "No corresponde preaprobar crédito ni gracia con la evidencia disponible. Pendiente: " + "; ".join(observaciones) + "."
        norma = (
            "Política de Microcrédito Preventivo y Reprogramación Preventiva: exige evaluación individual, "
            "trazabilidad y capacidad de pago. Cláusula de Ética y No Exclusión: el riesgo climático no permite rechazar ni encarecer automáticamente."
        )
    return DiagnosticoRiesgo(
        diagnostico_tecnico=diagnostico,
        instrumento_financiero_sugerido=instrumento,
        monto_o_condicion_aplicable=condicion,
        justificacion_normativa=norma,
        requiere_visita_campo=riesgo_fisico >= 0.60 or prob >= 0.45,
    )


def _acciones_catalogo(cliente: dict[str, Any], escenario: str, politicas: dict[str, Any]) -> list[MedidaOperativa]:
    catalogo = politicas["catalogo_acciones_preventivas"]
    actividad = str(cliente["actividad_economica"])
    acciones = catalogo[actividad]
    es_sequia = "Sequia" in escenario or (
        float(cliente.get("amenaza_sequia_escenario", 0)) > float(cliente.get("amenaza_inundacion_escenario", 0))
    )
    amenaza = "sequia_estres_hidrico" if es_sequia else "lluvias_inundaciones"
    preferidas = [a for a in acciones if a["amenaza"] == amenaza]
    seleccion = (preferidas + [a for a in acciones if a not in preferidas])[:3]
    # El catálogo de la Etapa 1 garantiza al menos dos por actividad.
    return [
        MedidaOperativa(
            accion=a["accion"], costo_estimado_pen=float(a["costo_estimado_pen"]),
            plazo_ejecucion_dias=int(a["tiempo_ejecucion_dias"]),
        )
        for a in seleccion
    ]


def _respaldo_continuidad(
    cliente: dict[str, Any], escenario: str, decision_riesgo: DiagnosticoRiesgo, politicas: dict[str, Any]
) -> PlanContinuidad:
    medidas = _acciones_catalogo(cliente, escenario, politicas)
    primera = medidas[0]
    mensaje = (
        f"Hola, ante el escenario de {escenario.replace('_', ' ')}, recomendamos empezar por: {primera.accion.lower()}. "
        f"Costo referencial S/ {primera.costo_estimado_pen:,.0f} y plazo {primera.plazo_ejecucion_dias} días. "
        "Podemos coordinar una visita preventiva para revisar tu negocio y alternativas de apoyo."
    )
    return PlanContinuidad(plan_accion_operativo=medidas, mensaje_whatsapp_cliente=mensaje[:450])


def _cliente_para_prompt(cliente: dict[str, Any]) -> dict[str, Any]:
    campos = [
        "id_cliente_hash", "actividad_economica", "tipo_local", "porcentaje_inventario_perecible",
        "saldo_capital_pen", "holgura_financiera", "dias_atraso_actual", "prob_estres_liquidez",
        "perdida_esperada_pen", "ivd_score", "nivel_confianza_alerta", "amenaza_inundacion_escenario",
        "amenaza_sequia_escenario", "nivel_riesgo",
    ]
    return {campo: cliente.get(campo) for campo in campos if campo in cliente}


def _crear_cliente_gemini() -> Any:
    """Inicializa Gemini solo cuando hace falta, conservando importación tolerante a fallos."""
    if load_dotenv:
        load_dotenv(RUTA_BASE / ".env")
    clave = os.getenv("GEMINI_API_KEY")
    if not clave:
        raise RuntimeError("GEMINI_API_KEY no está configurada")
    from google import genai
    return genai.Client(api_key=clave)


def _respuesta_gemini(cliente_api: Any, system_instruction: str, contenido: dict[str, Any], esquema: type[BaseModel]) -> BaseModel:
    """Solicita JSON schema estricto con el SDK oficial google-genai."""
    from google.genai import types
    respuesta = cliente_api.models.generate_content(
        model=MODELO_GEMINI,
        contents=json.dumps(contenido, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=esquema,
            temperature=0.0,
        ),
    )
    if not respuesta.text:
        raise RuntimeError("Gemini no devolvió contenido JSON")
    return esquema.model_validate_json(respuesta.text)


def _validar_decision_gemini(
    decision: DiagnosticoRiesgo, cliente: dict[str, Any], politicas: dict[str, Any]
) -> DiagnosticoRiesgo:
    """Evita que una respuesta correcta en forma incumpla la política en contenido."""
    cumple, _ = _hechos_suficientes_para_instrumento(cliente)
    texto = decision.instrumento_financiero_sugerido.lower()
    financiero = "microcrédito" in texto or "gracia" in texto or "reprogram" in texto
    prohibido = any(palabra in texto for palabra in ("rechaz", "bloque", "cobranza coactiva", "denegar"))
    if prohibido or (financiero and not cumple):
        return _respaldo_riesgo(cliente, politicas)
    return decision


def generar_expediente_preventivo(
    cliente: dict[str, Any], escenario: str, politicas: dict[str, Any], modo_respaldo: bool = False
) -> dict[str, Any]:
    """Ejecuta Agente 1 -> Agente 2 y retorna un expediente JSON serializable.

    Ante ausencia de red, cuota o clave, el mismo flujo cambia automáticamente al
    respaldo determinista y marca la fuente usada para auditoría.
    """
    respaldo_usado = modo_respaldo
    decision: DiagnosticoRiesgo
    plan: PlanContinuidad
    if not modo_respaldo:
        try:
            api = _crear_cliente_gemini()
            reglas_riesgo = {
                "politicas_riesgo_caja_trujillo": politicas["politicas_riesgo_caja_trujillo"],
                "niveles_incertidumbre": politicas["niveles_incertidumbre"],
            }
            sistema_riesgo = """Eres el Copiloto de Riesgo Crediticio de Caja Trujillo. Responde exclusivamente el JSON del esquema.
Usa únicamente los datos y políticas entregados. Cero alucinaciones: solo sugiere Microcrédito Preventivo o Periodo de Gracia si cada condición del JSON está demostrada y el nivel no es Baja. Si falta evidencia, indica gestión preventiva y validación humana. Está prohibido sugerir rechazo, bloqueo de líneas, encarecimiento automático o cobranza coactiva por riesgo climático. Distingue explícitamente amenaza física y fragilidad financiera. Cita el nombre de la política aplicable."""
            decision = _respuesta_gemini(api, sistema_riesgo, {"cliente": _cliente_para_prompt(cliente), "politicas": reglas_riesgo}, DiagnosticoRiesgo)
            decision = _validar_decision_gemini(decision, cliente, politicas)
            sistema_continuidad = """Eres el Asesor de Continuidad Operativa para microempresas de La Libertad. Responde exclusivamente JSON.
Elige 2 o 3 medidas solamente del catálogo recibido; conserva exactamente acción, costo y plazo. Redacta un WhatsApp empático en español peruano, máximo 450 caracteres, sin tecnicismos ni promesas de crédito. No inventes medidas, costos o plazos."""
            plan = _respuesta_gemini(api, sistema_continuidad, {
                "perfil": _cliente_para_prompt(cliente), "escenario": escenario,
                "decision_aprobada_agente_1": decision.model_dump(),
                "catalogo_acciones_preventivas": politicas["catalogo_acciones_preventivas"][cliente["actividad_economica"]],
            }, PlanContinuidad)
            # La validación final garantiza que medidas/costos no salgan del grounding.
            catalogo = _acciones_catalogo(cliente, escenario, politicas)
            permitidas = {(m.accion, m.costo_estimado_pen, m.plazo_ejecucion_dias) for m in catalogo}
            if any((m.accion, m.costo_estimado_pen, m.plazo_ejecucion_dias) not in permitidas for m in plan.plan_accion_operativo):
                plan = _respaldo_continuidad(cliente, escenario, decision, politicas)
        except Exception:
            respaldo_usado = True
            decision = _respaldo_riesgo(cliente, politicas)
            plan = _respaldo_continuidad(cliente, escenario, decision, politicas)
    else:
        decision = _respaldo_riesgo(cliente, politicas)
        plan = _respaldo_continuidad(cliente, escenario, decision, politicas)

    return {
        "id_cliente_hash": str(cliente["id_cliente_hash"]),
        "escenario_climatico": escenario,
        "modo_respaldo_usado": respaldo_usado,
        "copiloto_riesgo_crediticio": decision.model_dump(),
        "asesor_continuidad_operativa": plan.model_dump(),
    }


def procesar_top_prioritarios(
    ruta_csv: str = "cartera_evaluada.csv", top_n: int = 5, modo_respaldo: bool = False
) -> list[dict[str, Any]]:
    """Crea expedientes de los clientes priorizados y los exporta a JSON."""
    if top_n < 1:
        raise ValueError("top_n debe ser mayor que cero")
    ruta = Path(ruta_csv)
    if not ruta.is_absolute():
        ruta = RUTA_BASE / ruta
    cartera = pd.read_csv(ruta, encoding="utf-8-sig")
    requeridas = {"id_cliente_hash", "prioridad_intervencion", "ranking_urgencia", "escenario_climatico"}
    if faltantes := requeridas - set(cartera.columns):
        raise ValueError(f"cartera_evaluada.csv no contiene: {sorted(faltantes)}")
    prioritarios = cartera[cartera["prioridad_intervencion"].astype(str).str.lower().eq("true")]
    prioritarios = prioritarios.sort_values("ranking_urgencia").head(top_n)
    politicas = cargar_politicas()
    expedientes = [
        generar_expediente_preventivo(fila, str(fila["escenario_climatico"]), politicas, modo_respaldo)
        for fila in prioritarios.to_dict(orient="records")
    ]
    salida = RUTA_BASE / "expedientes_preventivos.json"
    with salida.open("w", encoding="utf-8") as archivo:
        json.dump(expedientes, archivo, ensure_ascii=False, indent=2)
    return expedientes


if __name__ == "__main__":
    expedientes_generados = procesar_top_prioritarios(top_n=3)
    if expedientes_generados:
        print(json.dumps(expedientes_generados[0], ensure_ascii=False, indent=2))
    else:
        print("No hay clientes con prioridad_intervencion=True para procesar.")
