"""Genera series SIMULADAS en datos/demo/ con las frecuencias y fechas de publicación
del catálogo, para probar la cadena completa sin acceso a las fuentes.
Ninguna cifra de esta carpeta es un dato real.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from datos import RAIZ, catalogo

DEMO = RAIZ / "datos" / "demo"


def main(semilla: int = 7) -> None:
    rng = np.random.default_rng(semilla)
    cat = catalogo()
    meses = pd.period_range("2000-01", "2026-09", freq="M")
    T = len(meses)
    ciclo = np.zeros(T)
    for t in range(1, T):
        ciclo[t] = 0.95 * ciclo[t - 1] + rng.normal(0, 0.35)
    ciclo[(meses >= "2008-10") & (meses <= "2009-06")] -= 3.0
    ciclo[(meses >= "2020-03") & (meses <= "2020-06")] -= 8.0
    energia = np.cumsum(rng.normal(0, 0.06, T))
    energia[meses >= "2026-03"] += 0.45
    tipos = np.zeros(T)
    tipos[0] = 4.0
    for t in range(1, T):
        tipos[t] = tipos[t - 1] + 0.03 * (2.2 - tipos[t - 1]) + 0.06 * ciclo[t - 1] / 3 + rng.normal(0, 0.08)
    tendencia = np.arange(T) / 12

    def mensual(sid, base, crec, beta_ciclo, beta_energia, ruido, estacional=0.0):
        estac = estacional * np.sin(2 * np.pi * meses.month.values / 12)
        lv = np.log(base) + crec * tendencia + beta_ciclo * ciclo / 100 + beta_energia * energia + estac \
            + np.cumsum(rng.normal(0, ruido, T))
        return pd.Series(np.exp(lv), index=meses.astype(str))

    series = {
        "hicp_es": mensual("hicp_es", 70, 0.021, 0.2, 0.08, 0.002, 0.01),
        "euribor3m": pd.Series(tipos, index=meses.astype(str)),
        "eurusd": mensual("eurusd", 1.0, 0.0, 0.3, -0.05, 0.02),
        "brent": pd.Series(np.exp(np.log(30) + 0.05 * tendencia + energia + 0.3 * ciclo / 10), index=meses.astype(str)),
        "gas": pd.Series(np.exp(np.log(3) + 0.04 * tendencia + 1.3 * energia + 0.2 * ciclo / 10), index=meses.astype(str)),
        "confianza_ind_ea": pd.Series(-4 + 2.5 * ciclo + rng.normal(0, 1.0, T), index=meses.astype(str)),
        "ipi_c29_eu": mensual("x", 90, 0.005, 3.0, -0.10, 0.008),
        "ipi_c29_es": mensual("x", 95, 0.000, 3.2, -0.12, 0.010),
        "aluminio": mensual("x", 1500, 0.02, 1.5, 0.30, 0.03),
        "ppi_plasticos_eu": mensual("x", 80, 0.015, 0.5, 0.40, 0.006),
        "export_8708_es": mensual("x", 600e6, 0.03, 3.0, -0.05, 0.015, 0.06),
        "ipi_c28_eu": mensual("x", 85, 0.008, 2.5, -0.05, 0.008),
        "ipi_c28_es": mensual("x", 90, 0.006, 2.8, -0.06, 0.010),
        "ip_aero_us": mensual("x", 80, 0.015, 0.8, -0.05, 0.010),
        "export_8456_8466_es": mensual("x", 90e6, 0.025, 2.5, -0.02, 0.02, 0.08),
        "ipi_c275_eu": mensual("x", 110, -0.004, 2.0, -0.08, 0.009),
        "confianza_consumidor_ea": pd.Series(-12 + 2.0 * ciclo - 6 * energia + rng.normal(0, 1.0, T), index=meses.astype(str)),
        "cobre": mensual("x", 1800, 0.05, 1.8, 0.25, 0.035),
        "import_cn_electrodomesticos_eu": mensual("x", 500e6, 0.05, 1.0, 0.0, 0.02, 0.07),
        "ipi_c27_es": mensual("x", 92, 0.004, 2.4, -0.08, 0.010),
        "construccion_ea": mensual("x", 105, 0.002, 1.5, -0.04, 0.006),
        "construccion_es": mensual("x", 160, -0.010, 2.5, -0.05, 0.010),
        "pernoctaciones_hoteleras_eu": mensual("x", 1.3e8, 0.012, 1.5, -0.05, 0.01, 0.35),
        "ppi_madera_es": mensual("x", 75, 0.02, 0.6, 0.15, 0.005),
        "ppi_acero_eu": mensual("x", 70, 0.02, 1.2, 0.35, 0.010),
    }
    trims = pd.period_range("2000Q1", "2026Q2", freq="Q")
    cq = pd.Series(ciclo, index=meses).groupby(meses.asfreq("Q")).mean().reindex(trims).values
    eq = pd.Series(energia, index=meses).groupby(meses.asfreq("Q")).mean().reindex(trims).values
    t_q = np.arange(len(trims)) / 4

    def trimestral(base, crec, beta, beta_e, ruido, estacional=0.0):
        lv = np.log(base) + crec * t_q + beta * cq / 100 + beta_e * eq + np.cumsum(rng.normal(0, ruido, len(trims))) \
            + estacional * np.sin(2 * np.pi * trims.quarter.values / 4)
        return pd.Series(np.exp(lv), index=trims.astype(str))

    series |= {
        "pib_ea": trimestral(2.4e6, 0.012, 1.0, -0.03, 0.002),
        "pib_es": trimestral(2.6e5, 0.016, 1.3, -0.04, 0.003),
        "pib_pv": trimestral(1.8e4, 0.015, 1.4, -0.04, 0.004),
        "coste_laboral_es": trimestral(70, 0.022, 0.2, 0.02, 0.004),
        "inversion_equipo_ea": trimestral(3.0e5, 0.012, 3.0, -0.05, 0.008),
        "utilizacion_capacidad_ea": pd.Series(80.5 + 0.9 * cq + rng.normal(0, 0.4, len(trims)), index=trims.astype(str)),
        "permisos_residenciales_es": trimestral(120, -0.02, 3.0, -0.05, 0.03, 0.05),
    }
    recortes = {"ipi_c29_eu": "2026-08", "ipi_c29_es": "2026-08", "ipi_c28_eu": "2026-08", "ipi_c28_es": "2026-08",
                "ipi_c275_eu": "2026-07", "ipi_c27_es": "2026-08", "export_8708_es": "2026-07",
                "export_8456_8466_es": "2026-07", "import_cn_electrodomesticos_eu": "2026-07",
                "construccion_ea": "2026-08", "construccion_es": "2026-07", "pernoctaciones_hoteleras_eu": "2026-08",
                "ppi_madera_es": "2026-08", "ppi_acero_eu": "2026-08", "ip_aero_us": "2026-08",
                "hicp_es": "2026-09", "confianza_consumidor_ea": "2026-09"}
    DEMO.mkdir(parents=True, exist_ok=True)
    for sid, s in series.items():
        if sid not in cat:
            continue
        if sid in recortes:
            s = s[s.index <= recortes[sid]]
        pd.DataFrame({"periodo": s.index, "valor": s.values}).to_csv(DEMO / f"{sid}.csv", index=False)
        (DEMO / f"{sid}.json").write_text(json.dumps({"id": sid, "etiqueta_fuente": "SIMULADO", "ultimo": s.index[-1]},
                                                     ensure_ascii=False), encoding="utf-8")
    print(f"{len(series)} series simuladas en {DEMO}")


if __name__ == "__main__":
    main()
