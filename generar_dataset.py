"""Genera una cartera sintética, anonimizada y reproducible para el MVP.

Dependencias permitidas: pandas y numpy (además de la biblioteca estándar).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


SEMILLA = 20261123
N_REGISTROS = 350
SALIDA = Path(__file__).with_name("cartera_sintetica_trujillo.csv")

# Los rangos son envolventes geográficos deliberadamente aproximados para datos
# sintéticos: representan zonas y no direcciones, predios ni hogares.
ZONAS = [
    {"zona": "El Porvenir - influencia Quebrada San Idelfonso", "id_agencia": "POR-02", "peso": 0.18,
     "lat": (-8.073, -8.050), "lon": (-79.010, -78.981), "inundacion": (0.76, 0.98), "sequia": (0.12, 0.31)},
    {"zona": "La Esperanza/Huanchaco - influencia Quebrada El León", "id_agencia": "ESP-04", "peso": 0.15,
     "lat": (-8.105, -8.062), "lon": (-79.080, -79.020), "inundacion": (0.70, 0.94), "sequia": (0.10, 0.29)},
    {"zona": "Ribera del Río Moche", "id_agencia": "TRU-01", "peso": 0.12,
     "lat": (-8.145, -8.105), "lon": (-79.015, -78.965), "inundacion": (0.72, 0.95), "sequia": (0.18, 0.42)},
    {"zona": "Valle de Virú", "id_agencia": "VIR-03", "peso": 0.16,
     "lat": (-8.430, -8.370), "lon": (-78.815, -78.735), "inundacion": (0.15, 0.38), "sequia": (0.70, 0.94)},
    {"zona": "Chao", "id_agencia": "VIR-03", "peso": 0.10,
     "lat": (-8.535, -8.455), "lon": (-78.735, -78.650), "inundacion": (0.12, 0.34), "sequia": (0.74, 0.97)},
    {"zona": "Laredo", "id_agencia": "TRU-01", "peso": 0.10,
     "lat": (-8.115, -8.070), "lon": (-78.995, -78.930), "inundacion": (0.25, 0.52), "sequia": (0.57, 0.84)},
    {"zona": "Centro de Trujillo", "id_agencia": "TRU-01", "peso": 0.11,
     "lat": (-8.125, -8.095), "lon": (-79.045, -79.010), "inundacion": (0.08, 0.30), "sequia": (0.05, 0.20)},
    {"zona": "Víctor Larco", "id_agencia": "TRU-01", "peso": 0.08,
     "lat": (-8.155, -8.125), "lon": (-79.075, -79.040), "inundacion": (0.10, 0.35), "sequia": (0.05, 0.22)},
]

ACTIVIDADES = {
    "Bodega y Abarrotes": {"ciiu": "4711", "peso": 0.23, "perecible": (0.18, 0.52), "locales": ["Alquilado", "Propio material noble", "Puesto de mercado"]},
    "Fabricación de Calzado": {"ciiu": "1520", "peso": 0.15, "perecible": (0.00, 0.06), "locales": ["Propio material noble", "Alquilado"]},
    "Agricultura Estacional": {"ciiu": "0113", "peso": 0.18, "perecible": (0.45, 0.95), "locales": ["Parcela agrícola"]},
    "Taller Mecánico/Metalmecánica": {"ciiu": "4520", "peso": 0.14, "perecible": (0.00, 0.04), "locales": ["Propio material noble", "Alquilado"]},
    "Restaurante y Alimentos": {"ciiu": "5610", "peso": 0.16, "perecible": (0.55, 0.96), "locales": ["Alquilado", "Propio material noble", "Puesto de mercado"]},
    "Comercio Textil": {"ciiu": "4771", "peso": 0.14, "perecible": (0.00, 0.08), "locales": ["Puesto de mercado", "Alquilado", "Propio material noble"]},
}


def elegir_actividad(rng: np.random.Generator, zona: str) -> str:
    """Sobrepondera agricultura en los valles sin eliminar los otros negocios."""
    nombres = list(ACTIVIDADES)
    pesos = np.array([ACTIVIDADES[n]["peso"] for n in nombres], dtype=float)
    if zona in {"Valle de Virú", "Chao", "Laredo"}:
        pesos[nombres.index("Agricultura Estacional")] *= 2.8
    return str(rng.choice(nombres, p=pesos / pesos.sum()))


def generar_registro(indice: int, rng: np.random.Generator) -> dict:
    pesos_zona = np.array([z["peso"] for z in ZONAS], dtype=float)
    zona = ZONAS[int(rng.choice(len(ZONAS), p=pesos_zona / pesos_zona.sum()))]
    actividad = elegir_actividad(rng, zona["zona"])
    perfil = ACTIVIDADES[actividad]
    saldo = float(rng.uniform(1500, 45000))
    plazo_meses = int(rng.choice([12, 18, 24, 30, 36], p=[0.10, 0.20, 0.34, 0.20, 0.16]))
    # Cuota aproximada de crédito amortizable: incluye una tasa mensual sintética.
    tasa_mensual = float(rng.uniform(0.018, 0.032))
    cuota = saldo * (tasa_mensual * (1 + tasa_mensual) ** plazo_meses) / ((1 + tasa_mensual) ** plazo_meses - 1)
    puntual = rng.random() < 0.79
    token = f"cliente-sintetico|{SEMILLA}|{indice:04d}"
    return {
        "id_cliente_hash": hashlib.sha256(token.encode("utf-8")).hexdigest()[:12],
        "id_agencia": zona["id_agencia"],
        "zona_referencia": zona["zona"],
        "latitud": round(float(rng.uniform(*zona["lat"])), 6),
        "longitud": round(float(rng.uniform(*zona["lon"])), 6),
        "actividad_economica": actividad,
        "codigo_ciiu": perfil["ciiu"],
        "tipo_local": str(rng.choice(perfil["locales"])),
        "porcentaje_inventario_perecible": round(float(rng.uniform(*perfil["perecible"])), 3),
        "saldo_capital_pen": round(saldo, 2),
        "cuota_mensual_pen": round(float(cuota), 2),
        "plazo_restante_meses": plazo_meses,
        "dias_atraso_actual": 0 if puntual else int(rng.integers(1, 16)),
        "holgura_financiera": round(float(rng.uniform(0.8, 2.5)), 2),
        "amenaza_inundacion_base": round(float(rng.uniform(*zona["inundacion"])), 3),
        "amenaza_sequia_base": round(float(rng.uniform(*zona["sequia"])), 3),
    }


def validar(df: pd.DataFrame) -> None:
    assert len(df) == N_REGISTROS
    assert df["id_cliente_hash"].nunique() == N_REGISTROS
    assert df["saldo_capital_pen"].between(1500, 45000).all()
    assert df["dias_atraso_actual"].between(0, 15).all()
    assert df["holgura_financiera"].between(0.8, 2.5).all()
    assert df[["amenaza_inundacion_base", "amenaza_sequia_base"]].apply(lambda s: s.between(0, 1).all()).all()
    # Prueba explícita de la regla de no determinismo solicitada.
    robustos_expuestos = df.loc[
        ((df["amenaza_inundacion_base"] > 0.75) | (df["amenaza_sequia_base"] > 0.75))
        & (df["holgura_financiera"] > 1.8)
        & (df["tipo_local"] == "Propio material noble")
    ]
    assert len(robustos_expuestos) >= 12, "Faltan casos robustos en zonas de alta amenaza"


def main() -> None:
    rng = np.random.default_rng(SEMILLA)
    registros = [generar_registro(i, rng) for i in range(N_REGISTROS)]
    df = pd.DataFrame(registros)
    # Garantiza una muestra sustancial de exposición alta con resiliencia física y financiera.
    candidatos = df.index[(df["amenaza_inundacion_base"] > 0.75) | (df["amenaza_sequia_base"] > 0.75)].tolist()
    for posicion in candidatos[:15]:
        df.loc[posicion, "tipo_local"] = "Propio material noble"
        df.loc[posicion, "holgura_financiera"] = round(float(rng.uniform(1.81, 2.5)), 2)
    validar(df)
    df.to_csv(SALIDA, index=False, encoding="utf-8-sig")
    print(f"Archivo generado: {SALIDA} ({len(df)} registros)")


if __name__ == "__main__":
    main()
