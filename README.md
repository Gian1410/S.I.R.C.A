# S.I.R.C.A.

## Sistema Inteligente de Resiliencia Climática Asistida

MVP para **Caja Trujillo** (La Libertad, Perú) que estima vulnerabilidad climática de una cartera de microempresas, prioriza intervenciones preventivas con optimización matemática y genera expedientes auditables con agentes de IA.

> Proyecto de innovación. Los datos son sintéticos y anonimizados; ninguna salida reemplaza la evaluación crediticia humana, la normativa vigente ni las políticas institucionales.

## Qué demuestra

- Simulación Monte Carlo de 500 iteraciones por cliente para tratar la incertidumbre del pronóstico.
- Cuatro escenarios: `Normal`, `Lluvias_Moderadas`, `El_Nino_Costero` y `Sequia_Severa`.
- Priorización mediante programación lineal con capacidad operativa y presupuesto limitado.
- Agentes con salida JSON validada, fundamento en políticas internas y respaldo offline determinista.
- Interfaz Streamlit para explorar mapa, KPIs, ranking de atención y planes preventivos.
- Principio explícito de no exclusión: el riesgo climático sirve para prevenir y acompañar, nunca para rechazar automáticamente un crédito.

## Arquitectura

```text
generar_dataset.py
        │
        ▼
cartera_sintetica_trujillo.csv ──┐
                                ├──► motor_riesgo_optimizacion.py
politicas_y_reglas.json ────────┘            │
                                             ▼
                                cartera_evaluada.csv
                                      │              │
                                      ▼              ▼
                    agentes_resiliencia.py      app_simulador.py
                              │                       │
                              ▼                       ▼
                  expedientes_preventivos.json   Demo interactiva
```

## Componentes

| Archivo | Función |
|---|---|
| `generar_dataset.py` | Crea 350 microempresas sintéticas, georreferenciadas y anonimizadas para La Libertad. |
| `politicas_y_reglas.json` | Fuente de verdad: sensibilidad sectorial, intervenciones, incertidumbre, ética y no exclusión. |
| `motor_riesgo_optimizacion.py` | Ejecuta simulación probabilística y optimiza la asignación preventiva. |
| `agentes_resiliencia.py` | Genera expedientes estructurados para analista y microempresario con Gemini o modo respaldo. |
| `app_simulador.py` | Aplicación Streamlit para la demostración en vivo. |

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/Gian1410/S.I.R.C.A.git
cd S.I.R.C.A
python -m venv .venv
```

Activa el entorno virtual:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Instala dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución rápida

Genera la cartera, calcula el escenario y abre la interfaz:

```bash
python generar_dataset.py
python motor_riesgo_optimizacion.py
streamlit run app_simulador.py
```

La aplicación se abre normalmente en `http://localhost:8501`.

## Gemini API y demostración offline

Para usar Gemini, crea un archivo `.env` en la raíz del proyecto:

```env
GEMINI_API_KEY=tu_clave_de_google_ai_studio
```

La clave no es necesaria para la demo. Activa **Modo Demo Offline (Sin API)** en la barra lateral: los agentes producirán el mismo formato JSON con reglas deterministas extraídas de `politicas_y_reglas.json`. También se activa automáticamente cuando falta la clave, hay un problema de conectividad o la API no responde.

## Interpretación de resultados

- `prob_estres_liquidez`: proporción de simulaciones donde la holgura estresada es menor que 1.0.
- `perdida_esperada_pen`: pérdida media simulada de cartera en riesgo.
- `perdida_peor_caso_p95_pen`: percentil 95 de pérdida simulada (VaR climático).
- `ivd_score`: Índice de Vulnerabilidad Dinámica entre 0 y 100.
- `prioridad_intervencion`: selección sujeta a capacidad, presupuesto y reglas de confianza.
- `ahorro_cartera_estimado_pen`: pérdida esperada mitigada, suponiendo una reducción de impacto del 65%.

Una amenaza alta no equivale a una pérdida automática: el modelo incorpora tipo de local, inventario perecible, holgura financiera y sensibilidad sectorial.

## Gobernanza y uso responsable

1. **No exclusión financiera:** está prohibido usar la puntuación climática para rechazar, encarecer automáticamente, bloquear líneas o iniciar cobranza coactiva.
2. **Intervención humana:** los instrumentos financieros requieren validación de elegibilidad, capacidad de pago, trazabilidad y aprobación humana.
3. **Incertidumbre explícita:** con confianza menor a 50%, el optimizador solo emite alertas informativas de costo cero.
4. **Privacidad:** se usan hashes truncados; no contiene nombres reales ni DNI.
5. **Trazabilidad:** las políticas, costos, plazos y criterios de elegibilidad se mantienen en un JSON auditable.

## Estructura de datos

Los CSV incluidos son sintéticos y pueden regenerarse. Las coordenadas representan zonas aproximadas de referencia —no direcciones ni predios— de Trujillo, El Porvenir, La Esperanza/Huanchaco, Moche, Virú, Chao, Laredo y Víctor Larco.

## Limitaciones del MVP

El modelo no consume aún pronósticos meteorológicos en tiempo real, información transaccional ni fuentes oficiales de alertas. Antes de una puesta en producción se requiere validación con riesgo, cumplimiento, datos, seguridad, legal y negocio; pruebas de sesgo; monitoreo; y controles de acceso.

## Licencia

Pendiente de definición por el equipo del proyecto.
