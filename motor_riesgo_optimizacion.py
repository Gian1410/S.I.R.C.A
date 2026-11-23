"""Motor de vulnerabilidad climática y priorización preventiva para Caja Trujillo.

Lee los artefactos de la Etapa 1 desde la misma carpeta y produce resultados
reproducibles. Las salidas son de apoyo preventivo: no deben utilizarse para
rechazar automáticamente solicitudes de crédito.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog


RUTA_BASE = Path(__file__).resolve().parent
RUTA_CARTERA = RUTA_BASE / "cartera_sintetica_trujillo.csv"
RUTA_POLITICAS = RUTA_BASE / "politicas_y_reglas.json"
RUTA_SALIDA = RUTA_BASE / "cartera_evaluada.csv"
ITERACIONES_MONTE_CARLO = 500
SEMILLA_MONTE_CARLO = 20261123
REDUCCION_IMPACTO_INTERVENCION = 0.65

ESCENARIOS: dict[str, tuple[float, float]] = {
    "Normal": (1.0, 1.0),
    "Lluvias_Moderadas": (1.45, 0.8),
    "El_Nino_Costero": (2.1, 0.6),
    "Sequia_Severa": (0.5, 2.0),
}

FACTOR_TIPO_LOCAL = {
    "Propio material noble": 0.70,
    "Alquilado": 0.95,
    "Puesto de mercado": 1.20,
    "Parcela agrícola": 1.15,
}


def cargar_insumos() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carga y valida los archivos producidos en la Etapa 1."""
    if not RUTA_CARTERA.exists():
        raise FileNotFoundError(f"No se encontró la cartera: {RUTA_CARTERA}. Ejecute primero generar_dataset.py.")
    if not RUTA_POLITICAS.exists():
        raise FileNotFoundError(f"No se encontró la política: {RUTA_POLITICAS}.")

    cartera = pd.read_csv(RUTA_CARTERA, encoding="utf-8-sig")
    with RUTA_POLITICAS.open("r", encoding="utf-8") as archivo:
        politicas = json.load(archivo)
    requeridas = {
        "id_cliente_hash", "actividad_economica", "tipo_local", "porcentaje_inventario_perecible",
        "saldo_capital_pen", "holgura_financiera", "amenaza_inundacion_base", "amenaza_sequia_base",
    }
    faltantes = requeridas - set(cartera.columns)
    if faltantes:
        raise ValueError(f"La cartera no contiene columnas requeridas: {sorted(faltantes)}")
    return cartera, politicas


def multiplicadores_escenario(escenario: str) -> tuple[float, float]:
    """Devuelve los multiplicadores de inundación y sequía para el escenario."""
    if escenario not in ESCENARIOS:
        raise ValueError(f"Escenario no válido: {escenario}. Opciones: {', '.join(ESCENARIOS)}")
    return ESCENARIOS[escenario]


def clasificar_confianza(confianza: float, niveles: dict[str, Any]) -> str:
    """Clasifica usando los límites declarados en niveles_incertidumbre del JSON."""
    if not 0.0 <= confianza <= 1.0:
        raise ValueError("confianza_pronostico debe estar entre 0.0 y 1.0")
    baja = re.findall(r"\d+(?:\.\d+)?", niveles["baja"]["rango_confianza"])
    media = re.findall(r"\d+(?:\.\d+)?", niveles["media"]["rango_confianza"])
    umbral_bajo = float(baja[0]) / 100 if baja else 0.50
    umbral_alto = float(media[-1]) / 100 if media else 0.75
    if confianza < umbral_bajo:
        return "Baja"
    if confianza <= umbral_alto:
        return "Media"
    return "Alta"


def obtener_costos_accion(cartera: pd.DataFrame, catalogo: dict[str, Any], amenaza_domina_lluvia: np.ndarray) -> np.ndarray:
    """Asigna el menor costo de la acción pertinente, leído del catálogo de políticas."""
    costos = np.zeros(len(cartera), dtype=float)
    for i, actividad in enumerate(cartera["actividad_economica"]):
        acciones = catalogo[actividad]
        amenaza = "lluvias_inundaciones" if amenaza_domina_lluvia[i] else "sequia_estres_hidrico"
        pertinentes = [a["costo_estimado_pen"] for a in acciones if a["amenaza"] == amenaza]
        # Si un rubro no tiene acción para la amenaza dominante, se usa la más barata
        # del catálogo, conservando una intervención concreta y verificable.
        costos[i] = min(pertinentes or [a["costo_estimado_pen"] for a in acciones])
    return costos


def simular_vulnerabilidad(
    cartera: pd.DataFrame,
    politicas: dict[str, Any],
    escenario: str,
    confianza_pronostico: float,
    iteraciones: int = ITERACIONES_MONTE_CARLO,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Ejecuta Monte Carlo y agrega las métricas probabilísticas por cliente."""
    mult_lluvia, mult_sequia = multiplicadores_escenario(escenario)
    nivel_confianza = clasificar_confianza(confianza_pronostico, politicas["niveles_incertidumbre"])
    rng = np.random.default_rng(SEMILLA_MONTE_CARLO)
    n = len(cartera)

    lluvia_esperada = np.clip(cartera["amenaza_inundacion_base"].to_numpy(float) * mult_lluvia, 0, 1)
    sequia_esperada = np.clip(cartera["amenaza_sequia_base"].to_numpy(float) * mult_sequia, 0, 1)
    # Con poca confianza se amplía la dispersión sin alterar la media del escenario.
    desviacion = 0.04 + 0.30 * (1.0 - confianza_pronostico)
    lluvia_simulada = np.clip(rng.normal(lluvia_esperada[:, None], desviacion, size=(n, iteraciones)), 0, 1)
    sequia_simulada = np.clip(rng.normal(sequia_esperada[:, None], desviacion, size=(n, iteraciones)), 0, 1)

    matriz = politicas["matriz_sensibilidad_sectorial"]
    sensibilidad_lluvia = cartera["actividad_economica"].map(
        lambda a: matriz[a]["lluvias_inundaciones"]["peso"]
    ).to_numpy(float)
    sensibilidad_sequia = cartera["actividad_economica"].map(
        lambda a: matriz[a]["sequia_estres_hidrico"]["peso"]
    ).to_numpy(float)
    factor_local = cartera["tipo_local"].map(FACTOR_TIPO_LOCAL).fillna(1.0).to_numpy(float)
    perecible = cartera["porcentaje_inventario_perecible"].clip(0, 1).to_numpy(float)

    # Impacto operativo como fracción de ingreso mensual: combinación de amenazas,
    # sensibilidad sectorial y mitigación/agravamiento físico. El tope evita pérdidas
    # superiores al 100% de ingresos en una sola ventana de simulación.
    impacto_bruto = (
        lluvia_simulada * sensibilidad_lluvia[:, None]
        + sequia_simulada * sensibilidad_sequia[:, None]
    ) / (sensibilidad_lluvia[:, None] + sensibilidad_sequia[:, None])
    factor_inventario = 0.75 + 0.50 * perecible
    impacto_ingresos = np.clip(impacto_bruto * factor_local[:, None] * factor_inventario[:, None], 0, 1)

    # La holgura equivale a la capacidad de absorber el golpe; se erosiona de forma
    # no lineal cuando el impacto crece y se penaliza levemente el atraso existente.
    holgura = cartera["holgura_financiera"].to_numpy(float)
    atraso = cartera.get("dias_atraso_actual", pd.Series(0, index=cartera.index)).to_numpy(float)
    holgura_estresada = holgura[:, None] / (1 + 1.65 * impacto_ingresos) - (atraso[:, None] / 100)
    estres = holgura_estresada < 1.0
    prob_estres = estres.mean(axis=1)

    saldo = cartera["saldo_capital_pen"].to_numpy(float)
    # Severidad condicional: combina daño operativo y déficit de holgura, y es cero
    # si no hay estrés. Así la pérdida esperada proviene de las 500 simulaciones.
    deficit = np.clip(1.0 - holgura_estresada, 0, 1)
    severidad = np.clip(0.15 + 0.65 * impacto_ingresos + 0.20 * deficit, 0, 1) * estres
    perdidas_simuladas = saldo[:, None] * severidad
    perdida_esperada = perdidas_simuladas.mean(axis=1)
    perdida_p95 = np.percentile(perdidas_simuladas, 95, axis=1)
    # Índice interpretable: probabilidad (55%), pérdida relativa (30%) y cola VaR (15%).
    ivd = np.clip(
        100 * (0.55 * prob_estres + 0.30 * (perdida_esperada / saldo) + 0.15 * (perdida_p95 / saldo)), 0, 100
    )
    nivel = np.select([ivd < 25, ivd < 50, ivd < 75], ["Bajo", "Moderado", "Alto"], default="Crítico")

    resultado = cartera.copy()
    resultado["escenario_climatico"] = escenario
    resultado["confianza_pronostico"] = confianza_pronostico
    resultado["nivel_confianza_alerta"] = nivel_confianza
    resultado["amenaza_inundacion_escenario"] = np.round(lluvia_esperada, 3)
    resultado["amenaza_sequia_escenario"] = np.round(sequia_esperada, 3)
    resultado["prob_estres_liquidez"] = np.round(prob_estres, 4)
    resultado["perdida_esperada_pen"] = np.round(perdida_esperada, 2)
    resultado["perdida_peor_caso_p95_pen"] = np.round(perdida_p95, 2)
    resultado["ivd_score"] = np.round(ivd, 2)
    resultado["nivel_riesgo"] = nivel
    return resultado, lluvia_esperada >= sequia_esperada


def optimizar_intervenciones(
    evaluada: pd.DataFrame,
    costos: np.ndarray,
    confianza_pronostico: float,
    capacidad_atencion: int,
    presupuesto_preventivo_pen: float,
) -> pd.DataFrame:
    """Prioriza con PL: maximiza pérdida mitigada, sujeta a capacidad y presupuesto."""
    if capacidad_atencion < 0 or presupuesto_preventivo_pen < 0:
        raise ValueError("capacidad_atencion y presupuesto_preventivo_pen no pueden ser negativos")
    resultado = evaluada.copy()
    beneficio = resultado["perdida_esperada_pen"].to_numpy(float) * REDUCCION_IMPACTO_INTERVENCION
    orden = np.lexsort((resultado["id_cliente_hash"].to_numpy(), -beneficio))
    ranking = np.empty(len(resultado), dtype=int)
    ranking[orden] = np.arange(1, len(resultado) + 1)
    resultado["costo_intervencion_estimado_pen"] = np.round(costos, 2)
    resultado["ranking_urgencia"] = ranking
    resultado["prioridad_intervencion"] = False
    resultado["ahorro_cartera_estimado_pen"] = 0.0
    resultado["tipo_intervencion"] = "Alerta informativa"

    # La política solo permite alertas de costo cero con confianza baja.
    if confianza_pronostico < 0.50 or capacidad_atencion == 0 or presupuesto_preventivo_pen == 0:
        return resultado

    # linprog minimiza: se niega el beneficio para maximizar ahorro. Variables 0..1;
    # la conversión conservadora requiere x prácticamente entero para no exceder límites.
    solucion = linprog(
        c=-beneficio,
        A_ub=np.vstack([np.ones(len(resultado)), costos]),
        b_ub=np.array([capacidad_atencion, presupuesto_preventivo_pen], dtype=float),
        bounds=(0, 1),
        method="highs",
    )
    if not solucion.success:
        raise RuntimeError(f"No se pudo resolver la priorización: {solucion.message}")

    seleccionados = solucion.x >= (1 - 1e-8)
    resultado.loc[seleccionados, "prioridad_intervencion"] = True
    resultado.loc[seleccionados, "ahorro_cartera_estimado_pen"] = np.round(beneficio[seleccionados], 2)
    resultado.loc[seleccionados, "tipo_intervencion"] = "Gestión preventiva priorizada"
    return resultado


def evaluar_y_optimizar_cartera(
    escenario: str,
    confianza_pronostico: float,
    capacidad_atencion: int = 35,
    presupuesto_preventivo_pen: float = 150000.0,
) -> pd.DataFrame:
    """Evalúa la cartera y devuelve una tabla lista para UI, agentes o exportación."""
    cartera, politicas = cargar_insumos()
    evaluada, domina_lluvia = simular_vulnerabilidad(cartera, politicas, escenario, confianza_pronostico)
    costos = obtener_costos_accion(evaluada, politicas["catalogo_acciones_preventivas"], domina_lluvia)
    return optimizar_intervenciones(
        evaluada, costos, confianza_pronostico, capacidad_atencion, presupuesto_preventivo_pen
    ).sort_values("ranking_urgencia").reset_index(drop=True)


def imprimir_resumen_ejecutivo(resultado: pd.DataFrame) -> None:
    """Imprime indicadores compactos para una prueba local del módulo."""
    seleccionados = resultado[resultado["prioridad_intervencion"]]
    print("\n=== RESUMEN EJECUTIVO: RESILIENCIA CLIMÁTICA ===")
    print(f"Clientes evaluados: {len(resultado)}")
    print(f"Escenario: {resultado['escenario_climatico'].iloc[0]}")
    print(f"Confianza de alerta: {resultado['nivel_confianza_alerta'].iloc[0]} ({resultado['confianza_pronostico'].iloc[0]:.0%})")
    print(f"Pérdida esperada total: S/ {resultado['perdida_esperada_pen'].sum():,.2f}")
    print(f"Clientes priorizados: {len(seleccionados)}")
    print(f"Costo preventivo seleccionado: S/ {seleccionados['costo_intervencion_estimado_pen'].sum():,.2f}")
    print(f"Ahorro estimado de cartera: S/ {seleccionados['ahorro_cartera_estimado_pen'].sum():,.2f}")
    print("Distribución de riesgo:")
    print(resultado["nivel_riesgo"].value_counts().reindex(["Bajo", "Moderado", "Alto", "Crítico"], fill_value=0).to_string())


if __name__ == "__main__":
    evaluacion = evaluar_y_optimizar_cartera(
        escenario="El_Nino_Costero",
        confianza_pronostico=0.82,
        capacidad_atencion=35,
        presupuesto_preventivo_pen=150000.0,
    )
    evaluacion.to_csv(RUTA_SALIDA, index=False, encoding="utf-8-sig")
    imprimir_resumen_ejecutivo(evaluacion)
    print(f"\nResultado exportado a: {RUTA_SALIDA}")
