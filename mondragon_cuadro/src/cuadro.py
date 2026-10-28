"""Construcción de condiciones a partir de anclas anuales y resumen de sendas en el cuadro."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bvar import Condicion

PESOS_MEDIA_ANUAL = [0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25]


def condicion_anual(i: int, tipo: str, anio: int, valor: float, sd: float,
                    trimestres_hist: pd.PeriodIndex, y_hist: np.ndarray, H: int,
                    etiqueta: str = "") -> Condicion | None:
    """Ancla anual sobre la variable i del núcleo.

    tipo "crecimiento": la variable está en 100*dlog trimestral y `valor` es la tasa de
    variación de la media anual en %. Se usa la aproximación logarítmica con pesos
    1-2-3-4-3-2-1 sobre las tasas trimestrales.
    tipo "nivel": la variable está en niveles y `valor` es su media anual.
    """
    ultimo = trimestres_hist[-1]
    if tipo == "crecimiento":
        objetivo = 100 * np.log1p(valor / 100)
        trims = pd.period_range(pd.Period(f"{anio - 1}Q2"), pd.Period(f"{anio}Q4"), freq="Q")
        pesos = PESOS_MEDIA_ANUAL
    elif tipo == "nivel":
        objetivo = valor
        trims = pd.period_range(pd.Period(f"{anio}Q1"), pd.Period(f"{anio}Q4"), freq="Q")
        pesos = [0.25] * 4
    else:
        raise ValueError(tipo)
    conocido = 0.0
    futuros: dict[tuple[int, int], float] = {}
    for q, w in zip(trims, pesos):
        if q <= ultimo:
            pos = trimestres_hist.get_loc(q)
            conocido += w * y_hist[pos, i]
        else:
            h = (q - ultimo).n - 1
            if h >= H:
                return None
            futuros[(h, i)] = w
    if not futuros:
        return None
    return Condicion(pesos=futuros, valor=objetivo - conocido, sd=sd, etiqueta=etiqueta)


def condicion_trimestre(i: int, h: int, valor: float, sd: float, etiqueta: str = "") -> Condicion:
    return Condicion(pesos={(h, i): 1.0}, valor=valor, sd=sd, etiqueta=etiqueta)


def a_niveles(nivel_ultimo: float, dlog: np.ndarray) -> np.ndarray:
    """dlog: (draws, H) en 100*dlog. Devuelve niveles (draws, H)."""
    return nivel_ultimo * np.exp(np.cumsum(dlog, axis=1) / 100)


def cuantiles(x: np.ndarray, qs=(0.05, 0.16, 0.5, 0.84, 0.95)) -> dict:
    v = np.quantile(x, qs, axis=0)
    return {f"p{int(q * 100):02d}": np.round(v[j], 3).tolist() for j, q in enumerate(qs)}


def resumen(nombre: str, tipo: str, hist: pd.Series, futuro: np.ndarray,
            trimestres_fut: pd.PeriodIndex, anios: list[int], nivel_hist: pd.Series | None = None) -> dict:
    """tipo "dlog": `hist` y `futuro` en 100*dlog trimestral, `nivel_hist` en niveles.
    tipo "nivel": `hist` y `futuro` en niveles (tipos, índices de confianza)."""
    if tipo == "dlog":
        niveles_fut = a_niveles(float(nivel_hist.iloc[-1]), futuro)
        serie_hist = nivel_hist
    else:
        niveles_fut = futuro
        serie_hist = hist
    D = niveles_fut.shape[0]
    todos = pd.PeriodIndex(list(serie_hist.index) + list(trimestres_fut), freq="Q")
    hist_rep = np.repeat(serie_hist.values[None, :], D, axis=0)
    completo = pd.DataFrame(np.hstack([hist_rep, niveles_fut]).T, index=todos)
    anual = {}
    for a in anios:
        actual = completo[completo.index.year == a]
        previo = completo[completo.index.year == a - 1]
        if len(actual) < 4:
            continue
        if tipo == "dlog":
            if len(previo) < 4:
                continue
            g = 100 * (actual.mean(axis=0).values / previo.mean(axis=0).values - 1)
        else:
            g = actual.mean(axis=0).values
        anual[str(a)] = cuantiles(g)
    if tipo == "dlog":
        yoy_hist = 100 * (serie_hist / serie_hist.shift(4) - 1)
        yoy_fut = 100 * (completo.values[len(serie_hist):] / completo.values[len(serie_hist) - 4: -4] - 1)
        trimestral = cuantiles(yoy_fut.T)
        hist_out = yoy_hist.dropna().iloc[-16:]
    else:
        trimestral = cuantiles(niveles_fut)
        hist_out = serie_hist.dropna().iloc[-16:]
    return {
        "anual": anual,
        "trimestral": trimestral,
        "trimestres": [str(q) for q in trimestres_fut],
        "historia": {"trimestres": [str(q) for q in hist_out.index], "valores": np.round(hist_out.values, 3).tolist()},
    }
