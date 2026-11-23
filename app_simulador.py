"""Interfaz Streamlit de S.I.R.C.A. para la demostración de Caja Trujillo.

Ejecutar después de generar la cartera y del motor de riesgo:
    streamlit run app_simulador.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from agentes_resiliencia import generar_expediente_preventivo
from motor_riesgo_optimizacion import evaluar_y_optimizar_cartera


RUTA_BASE = Path(__file__).resolve().parent
RUTA_POLITICAS = RUTA_BASE / "politicas_y_reglas.json"
COLORES_RIESGO = {
    "Bajo": "#22c55e",
    "Moderado": "#eab308",
    "Alto": "#f97316",
    "Crítico": "#dc2626",
}


st.set_page_config(
    page_title="S.I.R.C.A. | Caja Trujillo",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cargar_politicas() -> dict[str, Any]:
    """Carga la fuente de verdad de gobernanza una sola vez por sesión."""
    with RUTA_POLITICAS.open("r", encoding="utf-8") as archivo:
        return json.load(archivo)


@st.cache_data(show_spinner="Simulando incertidumbre climática y optimizando cartera...")
def simular_cartera(
    escenario: str,
    confianza_pronostico: float,
    capacidad_atencion: int,
    presupuesto_preventivo_pen: float,
) -> pd.DataFrame:
    """Memoiza Monte Carlo + programación lineal para reruns instantáneos."""
    return evaluar_y_optimizar_cartera(
        escenario=escenario,
        confianza_pronostico=confianza_pronostico,
        capacidad_atencion=capacidad_atencion,
        presupuesto_preventivo_pen=presupuesto_preventivo_pen,
    )


def nivel_confianza_texto(confianza: float) -> tuple[str, str]:
    if confianza < 0.50:
        return "Baja", "⚪ Solo alertas informativas; no se asignan intervenciones financieras."
    if confianza <= 0.75:
        return "Media", "🟡 Se permite contacto preventivo y evaluación individual."
    return "Alta", "🟢 Se permite priorizar evaluación acelerada e intervención documentada."


def formatear_soles(valor: float) -> str:
    return f"S/ {valor:,.0f}"


def buscar_ejemplo_no_determinismo(cartera: pd.DataFrame) -> pd.Series | None:
    """Encuentra evidencia visual de que amenaza no es igual a pérdida automática."""
    amenaza_alta = (
        (cartera["amenaza_inundacion_escenario"] > 0.75)
        | (cartera["amenaza_sequia_escenario"] > 0.75)
    )
    candidatos = cartera[
        amenaza_alta
        & (~cartera["prioridad_intervencion"])
        & (cartera["holgura_financiera"] > 1.8)
        & (cartera["tipo_local"] == "Propio material noble")
    ].sort_values("ranking_urgencia")
    return None if candidatos.empty else candidatos.iloc[0]


def crear_mapa_riesgo(cartera: pd.DataFrame):
    """Crea el mapa usando la API disponible en la versión instalada de Plotly.

    Plotly 6 usa ``scatter_map`` (MapLibre); versiones anteriores usan
    ``scatter_mapbox``. Ambas modalidades emplean OpenStreetMap y no requieren
    token externo.
    """
    argumentos = dict(
        data_frame=cartera,
        lat="latitud",
        lon="longitud",
        color="nivel_riesgo",
        size="saldo_capital_pen",
        size_max=23,
        color_discrete_map=COLORES_RIESGO,
        category_orders={"nivel_riesgo": ["Bajo", "Moderado", "Alto", "Crítico"]},
        hover_name="id_cliente_hash",
        hover_data={
            "actividad_economica": True, "tipo_local": True, "holgura_financiera": ":.2f",
            "prob_estres_liquidez": ":.1%", "prioridad_intervencion": True,
            "latitud": False, "longitud": False, "saldo_capital_pen": ":,.0f",
        },
        zoom=10.5,
        center={"lat": -8.11, "lon": -79.03},
        height=510,
    )
    if hasattr(px, "scatter_map"):
        figura = px.scatter_map(**argumentos)
        figura.update_layout(map_style="open-street-map")
    elif hasattr(px, "scatter_mapbox"):
        figura = px.scatter_mapbox(**argumentos)
        figura.update_layout(mapbox_style="open-street-map")
    else:
        raise RuntimeError("La versión de Plotly instalada no incluye scatter_map ni scatter_mapbox. Actualice Plotly.")
    figura.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend_title_text="Nivel")
    return figura


def limpiar_expediente_previo(llave_actual: tuple[Any, ...]) -> None:
    """Evita mostrar un expediente de un escenario o cliente anterior."""
    if st.session_state.get("llave_expediente") != llave_actual:
        st.session_state.pop("expediente_preventivo", None)
        st.session_state["llave_expediente"] = llave_actual


def mostrar_agentes(expediente: dict[str, Any]) -> None:
    """Renderiza de manera separada las salidas estrictamente estructuradas."""
    riesgo = expediente["copiloto_riesgo_crediticio"]
    continuidad = expediente["asesor_continuidad_operativa"]
    col_analista, col_cliente = st.columns(2)

    with col_analista:
        st.subheader("Agente 1 · Copiloto de Riesgo")
        st.caption("Vista para el analista de Caja Trujillo")
        st.info(riesgo["diagnostico_tecnico"])
        st.markdown("**Instrumento / contención**")
        st.write(riesgo["instrumento_financiero_sugerido"])
        st.markdown("**Monto o condición aplicable**")
        st.write(riesgo["monto_o_condicion_aplicable"])
        st.markdown("**Cita de política interna**")
        st.caption(riesgo["justificacion_normativa"])
        visita = "Sí, requiere verificación" if riesgo["requiere_visita_campo"] else "No inmediata"
        st.write(f"**Visita de campo:** {visita}")
        st.success("Cumplimiento: Política de No Exclusión Financiera")

    with col_cliente:
        st.subheader("Agente 2 · Asesor de Continuidad")
        st.caption("Vista para el microempresario")
        acciones = pd.DataFrame(continuidad["plan_accion_operativo"])
        acciones = acciones.rename(columns={
            "accion": "Acción preventiva",
            "costo_estimado_pen": "Inversión (S/)",
            "plazo_ejecucion_dias": "Plazo (días)",
        })
        if "Inversión (S/)" in acciones:
            acciones["Inversión (S/)"] = acciones["Inversión (S/)"].map(formatear_soles)
        st.dataframe(acciones, hide_index=True, use_container_width=True)
        st.markdown("**Mensaje WhatsApp listo para enviar**")
        st.success(continuidad["mensaje_whatsapp_cliente"])


def main() -> None:
    try:
        politicas = cargar_politicas()
    except FileNotFoundError:
        st.error("No se encontró politicas_y_reglas.json. Ejecute o copie primero los archivos de las etapas anteriores.")
        st.stop()

    st.title("S.I.R.C.A. · Simulador de Resiliencia Climática")
    st.caption("Caja Trujillo · La Libertad | Simulación probabilística, optimización preventiva y agentes auditables")

    with st.sidebar:
        st.header("Panel de escenarios")
        escenario = st.selectbox(
            "Escenario climático",
            ["Normal", "Lluvias_Moderadas", "El_Nino_Costero", "Sequia_Severa"],
            index=2,
        )
        confianza = st.slider("Confianza del pronóstico", min_value=0.35, max_value=0.95, value=0.82, step=0.01)
        nivel, descripcion = nivel_confianza_texto(confianza)
        st.markdown(f"**Nivel de incertidumbre: {nivel}**")
        st.caption(descripcion)
        st.divider()
        capacidad = st.slider("Capacidad de atención semanal", min_value=10, max_value=60, value=35, step=1)
        presupuesto = st.slider(
            "Presupuesto preventivo (S/)", min_value=50_000, max_value=300_000,
            value=150_000, step=5_000,
        )
        modo_respaldo = st.toggle("Modo Demo Offline (Sin API)", value=False)
        if modo_respaldo:
            st.caption("Los expedientes usarán reglas deterministas del JSON de políticas.")

    try:
        cartera = simular_cartera(escenario, confianza, capacidad, float(presupuesto))
    except Exception as error:
        st.error(f"No se pudo calcular la cartera: {error}")
        st.info("Verifique que cartera_sintetica_trujillo.csv exista y que pandas, numpy y scipy estén instalados.")
        st.stop()

    priorizados = cartera[cartera["prioridad_intervencion"]].copy()
    total_esperado = float(cartera["perdida_esperada_pen"].sum())
    total_var = float(cartera["perdida_peor_caso_p95_pen"].sum())
    ahorro = float(priorizados["ahorro_cartera_estimado_pen"].sum())
    metricas = st.columns(4)
    metricas[0].metric("Cartera en riesgo esperado", formatear_soles(total_esperado))
    metricas[1].metric("Peor escenario climático · VaR P95", formatear_soles(total_var))
    metricas[2].metric("Clientes priorizados", f"{len(priorizados)} / {capacidad}")
    metricas[3].metric("Ahorro estimado en mora", formatear_soles(ahorro))

    st.divider()
    mapa_col, ranking_col = st.columns([3, 2])
    with mapa_col:
        st.subheader("Mapa de exposición y vulnerabilidad")
        figura = crear_mapa_riesgo(cartera)
        st.plotly_chart(figura, use_container_width=True)

    with ranking_col:
        st.subheader("Ranking de intervención óptima")
        columnas = ["id_cliente_hash", "actividad_economica", "ivd_score", "prob_estres_liquidez", "ahorro_cartera_estimado_pen"]
        tabla = priorizados.sort_values("ranking_urgencia")[columnas].copy()
        tabla["prob_estres_liquidez"] = tabla["prob_estres_liquidez"].map(lambda x: f"{x:.1%}")
        tabla["ahorro_cartera_estimado_pen"] = tabla["ahorro_cartera_estimado_pen"].map(formatear_soles)
        st.dataframe(tabla, hide_index=True, use_container_width=True, height=325)
        ejemplo = buscar_ejemplo_no_determinismo(cartera)
        if ejemplo is not None:
            amenaza = max(float(ejemplo["amenaza_inundacion_escenario"]), float(ejemplo["amenaza_sequia_escenario"]))
            st.info(
                f"**No-determinismo en acción:** {ejemplo['id_cliente_hash']} tiene amenaza {amenaza:.0%}, "
                f"holgura {ejemplo['holgura_financiera']:.2f} y local propio de material noble; no fue priorizado. "
                "La amenaza física no implica pérdida automática."
            )
        else:
            st.caption("En este escenario no hay un caso robusto no priorizado; cambie escenario o restricciones para explorar el principio de no-determinismo.")

    st.divider()
    st.subheader("Centro de ejecución multi-agente")
    if priorizados.empty:
        st.warning("No hay clientes priorizados con las restricciones actuales. Ajuste la confianza, capacidad o presupuesto.")
        return

    opciones: dict[str, str] = {}
    for _, fila in priorizados.sort_values("ranking_urgencia").iterrows():
        etiqueta = f"#{int(fila['ranking_urgencia'])} · {fila['id_cliente_hash']} · {fila['actividad_economica']} · {formatear_soles(float(fila['saldo_capital_pen']))}"
        opciones[etiqueta] = str(fila["id_cliente_hash"])
    etiqueta_elegida = st.selectbox("Cliente priorizado", list(opciones))
    id_elegido = opciones[etiqueta_elegida]
    cliente = priorizados.loc[priorizados["id_cliente_hash"].astype(str) == id_elegido].iloc[0].to_dict()
    llave = (id_elegido, escenario, confianza, capacidad, presupuesto, modo_respaldo)
    limpiar_expediente_previo(llave)

    if st.button("Generar Plan Preventivo con Agentes IA", type="primary"):
        with st.spinner("Generando expediente auditado..."):
            st.session_state["expediente_preventivo"] = generar_expediente_preventivo(
                cliente=cliente, escenario=escenario, politicas=politicas, modo_respaldo=modo_respaldo
            )
    expediente = st.session_state.get("expediente_preventivo")
    if expediente:
        fuente = "respaldo determinista" if expediente["modo_respaldo_usado"] else "Gemini con salida JSON validada"
        st.caption(f"Fuente del expediente: {fuente}.")
        mostrar_agentes(expediente)


if __name__ == "__main__":
    main()
