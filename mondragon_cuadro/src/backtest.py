"""Evaluación fuera de muestra del núcleo sin condicionar: orígenes anuales móviles,
previsión mediana a 1, 4 y 8 trimestres frente a un AR(1) por variable.
Contrasta lambda1 en {0.1, 0.2, 0.3}. Escribe salida/backtest.json.

Uso: python src/backtest.py [--demo] [--rapido]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from bvar import PriorBVAR, estimar, prever
from datos import RAIZ, a_trimestral, catalogo, leer_bruta, modelo, transformar

HORIZONTES = (1, 4, 8)


def _ar1(serie: np.ndarray, H: int) -> np.ndarray:
    x, y = serie[:-1], serie[1:]
    A = np.column_stack([np.ones_like(x), x])
    c, r = np.linalg.lstsq(A, y, rcond=None)[0]
    out, ult = [], serie[-1]
    for _ in range(H):
        ult = c + r * ult
        out.append(ult)
    return np.array(out)


def main(demo: bool = False, rapido: bool = False) -> dict:
    cat, mod = catalogo(), modelo()
    carpeta = RAIZ / "datos" / ("demo" if demo else "brutos")
    cfg = mod["nucleo"]
    nucleo, cols = [], {}
    for v in cfg["variables"]:
        s = leer_bruta(v, carpeta)
        if s is None:
            continue
        q, _ = a_trimestral(s, cat[v])
        cols[v] = transformar(q, cat[v])
        nucleo.append(v)
    panel = pd.DataFrame(cols)[nucleo]
    panel = panel[panel.index >= pd.Period(mod["muestra"]["inicio"], freq="Q")].dropna()
    excl_all = np.isin(panel.index.astype(str), mod["muestra"]["excluir"])
    origenes = [o for o in pd.period_range("2012Q4", panel.index[-1] - max(HORIZONTES), freq="Q") if o.quarter == 4]
    if rapido:
        origenes = origenes[::3]
    lambdas = [0.2] if rapido else [0.1, 0.2, 0.3]
    errores = {lam: {h: [] for h in HORIZONTES} for lam in lambdas}
    errores_ar = {h: [] for h in HORIZONTES}
    for o in origenes:
        pos = panel.index.get_loc(o)
        y = panel.values[: pos + 1]
        excl = excl_all[: pos + 1]
        real = panel.values[pos + 1: pos + 1 + max(HORIZONTES)]
        if np.isin(panel.index[pos + 1: pos + 1 + max(HORIZONTES)].astype(str), mod["muestra"]["excluir"]).any():
            continue
        medias = [float(pd.Series(y[:, i]).mean()) if cfg["variables"][v]["media"] == "muestral"
                  else float(cfg["variables"][v]["media"]) for i, v in enumerate(nucleo)]
        for lam in lambdas:
            prior = PriorBVAR(np.array(medias), np.array([cfg["variables"][v]["sd"] for v in nucleo], dtype=float),
                              np.array([cfg["variables"][v]["propio"] for v in nucleo], dtype=float), lambda1=lam,
                              lambda2=cfg["lambda2"])
            post = estimar(y, nucleo, cfg["retardos"], prior, excluir=excl, n_draws=300, burn=200, semilla=5)
            med = np.median(prever(post, y, max(HORIZONTES), semilla=6), axis=0)
            for h in HORIZONTES:
                errores[lam][h].append(med[h - 1] - real[h - 1])
        ar = np.column_stack([_ar1(y[:, i], max(HORIZONTES)) for i in range(len(nucleo))])
        for h in HORIZONTES:
            errores_ar[h].append(ar[h - 1] - real[h - 1])
    rmse = lambda e: np.sqrt(np.mean(np.square(np.array(e)), axis=0))  # noqa: E731
    res = {"origenes": [str(o) for o in origenes], "variables": nucleo, "horizontes": list(HORIZONTES), "por_lambda": {}}
    for lam in lambdas:
        res["por_lambda"][str(lam)] = {
            str(h): {"rmse_bvar": np.round(rmse(errores[lam][h]), 3).tolist(),
                     "rmse_ar1": np.round(rmse(errores_ar[h]), 3).tolist(),
                     "ratio": np.round(rmse(errores[lam][h]) / rmse(errores_ar[h]), 3).tolist()} for h in HORIZONTES}
    medias_ratio = {lam: float(np.mean([np.mean(res["por_lambda"][str(lam)][str(h)]["ratio"]) for h in HORIZONTES]))
                    for lam in lambdas}
    res["lambda_recomendado"] = min(medias_ratio, key=medias_ratio.get)
    res["ratio_medio"] = {str(k): round(v, 3) for k, v in medias_ratio.items()}
    (RAIZ / "salida").mkdir(exist_ok=True)
    nombre = "backtest_demo.json" if demo else "backtest.json"
    (RAIZ / "salida" / nombre).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(origenes)} orígenes; ratio RMSE BVAR/AR(1) medio por lambda: {res['ratio_medio']}; "
          f"recomendado {res['lambda_recomendado']}")
    return res


if __name__ == "__main__":
    main(demo="--demo" in sys.argv, rapido="--rapido" in sys.argv)
