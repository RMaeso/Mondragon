"""Lectura de la caché de series, paso a trimestral y transformaciones."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parents[1]
BRUTOS = RAIZ / "datos" / "brutos"


def catalogo() -> dict:
    return yaml.safe_load((RAIZ / "config" / "series.yaml").read_text(encoding="utf-8"))


def modelo() -> dict:
    return yaml.safe_load((RAIZ / "config" / "modelo.yaml").read_text(encoding="utf-8"))


def leer_bruta(sid: str, carpeta: Path = BRUTOS) -> pd.Series | None:
    f = carpeta / f"{sid}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, dtype={"periodo": str})
    s = pd.Series(df["valor"].values, index=df["periodo"].values, dtype=float).dropna()
    return s


def normalizar_periodo(p: str, frecuencia: str) -> str:
    """Convierte 2026M08, 2026-08, 2026-08-01 -> 2026-08 y 2026-Q2, 2026Q2 -> 2026Q2."""
    p = str(p).strip()
    if frecuencia == "M":
        p = p.replace("M", "-")[:7]
        return p
    return p.replace("-Q", "Q")


def _indice(periodos: pd.Index, frecuencia: str) -> pd.PeriodIndex:
    return pd.PeriodIndex([normalizar_periodo(p, frecuencia) for p in periodos], freq=frecuencia)


def a_trimestral(s: pd.Series, ficha: dict) -> tuple[pd.Series, dict]:
    """Devuelve la serie trimestral de trimestres completos y, si el último trimestre
    está incompleto, su estimación parcial con el número de meses disponibles."""
    frec = ficha["frecuencia"]
    s = s.copy()
    s.index = _indice(s.index, frec)
    s = s.sort_index()
    parcial = {}
    if frec == "Q":
        return s, parcial
    q = s.index.asfreq("Q")
    g = s.groupby(q)
    n = g.size()
    agg = g.sum() if ficha["agregacion"] == "suma" else g.mean()
    completos = agg[n == 3]
    ultimo = agg.index[-1]
    if n.iloc[-1] < 3 and ficha["agregacion"] == "media":
        parcial = {"trimestre": str(ultimo), "valor": float(agg.iloc[-1]), "meses": int(n.iloc[-1])}
    return completos, parcial


def desestacionalizar(s: pd.Series) -> pd.Series:
    """Descomposición multiplicativa clásica con media móvil centrada 2x4."""
    ls = np.log(s)
    tendencia = ls.rolling(4, center=True).mean().rolling(2).mean().shift(-1)
    estacional = (ls - tendencia).groupby(ls.index.quarter).mean()
    estacional -= estacional.mean()
    return np.exp(ls - ls.index.quarter.map(estacional).values)


def transformar(s: pd.Series, ficha: dict) -> pd.Series:
    if ficha.get("ajuste") == "NSA_DESEST":
        s = desestacionalizar(s)
    if ficha["transformacion"] == "dlog":
        return 100 * np.log(s).diff()
    return s


def nivel_para_salida(s: pd.Series, ficha: dict) -> pd.Series:
    if ficha.get("ajuste") == "NSA_DESEST":
        return desestacionalizar(s)
    return s
