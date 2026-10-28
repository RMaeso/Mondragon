import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bvar import PriorBVAR, estimar, prever, Condicion  # noqa: E402
from cuadro import condicion_anual, resumen  # noqa: E402
from satelites import estimar_satelite, simular_satelite  # noqa: E402


def simular_var(T=400, semilla=3):
    rng = np.random.default_rng(semilla)
    mu = np.array([0.4, 0.5, 2.0])
    B1 = np.array([[0.4, 0.1, 0.0], [0.2, 0.3, 0.0], [0.1, 0.05, 0.85]])
    B2 = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.05]])
    S = np.array([[0.30, 0.05, 0.02], [0.05, 0.20, 0.01], [0.02, 0.01, 0.10]])
    L = np.linalg.cholesky(S)
    y = np.zeros((T, 3)) + mu
    for t in range(2, T):
        y[t] = mu + B1 @ (y[t - 1] - mu) + B2 @ (y[t - 2] - mu) + L @ rng.standard_normal(3)
    return y, mu, B1


def test_recupera_parametros():
    y, mu, B1 = simular_var()
    prior = PriorBVAR(media_mu=np.zeros(3), sd_mu=np.full(3, 5.0),
                      media_propio_retardo=np.array([0.0, 0.0, 0.9]), lambda1=0.5)
    post = estimar(y, ["a", "b", "c"], p=2, prior=prior, n_draws=600, burn=200, semilla=1)
    assert np.allclose(post.mu.mean(0), mu, atol=0.25), post.mu.mean(0)
    B1_est = post.B.mean(0)[:3, :].T
    assert np.abs(B1_est - B1).max() < 0.15, B1_est


def test_condicion_dura_se_cumple():
    y, _, _ = simular_var(T=200)
    prior = PriorBVAR(media_mu=np.array([0.4, 0.5, 2.0]), sd_mu=np.full(3, 0.5),
                      media_propio_retardo=np.array([0.0, 0.0, 0.9]))
    post = estimar(y, ["a", "b", "c"], p=2, prior=prior, n_draws=200, burn=100, semilla=2)
    c1 = Condicion(pesos={(0, 0): 1.0}, valor=1.5)
    c2 = Condicion(pesos={(2, 2): 0.5, (3, 2): 0.5}, valor=3.0)
    sendas = prever(post, y, H=8, condiciones=[c1, c2], semilla=4)
    assert np.allclose(sendas[:, 0, 0], 1.5, atol=1e-8)
    assert np.allclose(0.5 * sendas[:, 2, 2] + 0.5 * sendas[:, 3, 2], 3.0, atol=1e-8)
    libres = prever(post, y, H=8, semilla=4)
    assert libres[:, 0, 0].std() > 0.1


def test_ancla_anual_aproxima_media_anual():
    rng = np.random.default_rng(0)
    trims = pd.period_range("2020Q1", "2026Q2", freq="Q")
    d = rng.normal(0.4, 0.5, len(trims) + 6)
    d_hist = d[: len(trims)]
    y_hist = d_hist[:, None]
    c = condicion_anual(0, "crecimiento", 2026, 2.0, 0.0, trims, y_hist, H=6)
    fut = np.zeros((6, 1))
    exceso = c.valor / sum(c.pesos.values())
    for (h, _), w in c.pesos.items():
        fut[h, 0] = exceso
    serie = np.concatenate([d_hist, fut[:, 0]])
    nivel = 100 * np.exp(np.cumsum(serie) / 100)
    idx = pd.period_range("2020Q1", periods=len(serie), freq="Q")
    s = pd.Series(nivel, index=idx)
    g = 100 * (s[s.index.year == 2026].mean() / s[s.index.year == 2025].mean() - 1)
    assert abs(g - 2.0) < 0.05, g


def test_satelite_y_resumen():
    rng = np.random.default_rng(5)
    T = 160
    X = rng.normal(0, 1, (T, 2))
    s = np.zeros(T)
    for t in range(1, T):
        s[t] = 0.2 + 0.3 * s[t - 1] + 0.8 * X[t, 0] + rng.normal(0, 0.3)
    trims = pd.period_range("1987Q1", periods=T, freq="Q")
    sat = estimar_satelite("s", s, X, [0], trims.quarter.values, n_draws=500)
    assert abs(sat.coef[:, 2].mean() - 0.8) < 0.1
    sendas = rng.normal(0, 1, (300, 8, 2))
    fut_q = pd.period_range(trims[-1] + 1, periods=8, freq="Q")
    Xs = np.concatenate([np.repeat(X[-1][None, None, :], 300, axis=0), sendas], axis=1)
    sim = simular_satelite(sat, s[-1], Xs, fut_q.quarter.values)
    cero = simular_satelite(sat, 0.0, np.zeros((2000, 41, 2)), np.ones(40, dtype=int))
    media_larga = 0.2 / (1 - 0.3)
    assert abs(cero[:, -1].mean() - media_larga) < 0.06, cero[:, -1].mean()
    nivel = pd.Series(100 * np.exp(np.cumsum(s) / 100), index=trims)
    r = resumen("s", "dlog", pd.Series(s, index=trims), sim, fut_q, [2027, 2028], nivel_hist=nivel)
    assert "trimestral" in r and len(r["trimestral"]["p50"]) == 8


def test_jsonstat_y_periodos():
    from descarga import _jsonstat
    from datos import normalizar_periodo
    js = {"id": ["geo", "product", "time"], "size": [1, 2, 3], "label": "prueba",
          "dimension": {"time": {"category": {"index": {"2026-07": 0, "2026-08": 1, "2026-09": 2}}}},
          "value": {"0": 1.0, "1": 2.0, "2": 3.0, "3": 10.0, "5": 30.0}}
    s, etq = _jsonstat(js)
    assert s.to_dict() == {"2026-07": 11.0, "2026-08": 2.0, "2026-09": 33.0}
    assert normalizar_periodo("2026M08", "M") == "2026-08"
    assert normalizar_periodo("2026-08-01", "M") == "2026-08"
    assert normalizar_periodo("2026-Q2", "Q") == "2026Q2"


if __name__ == "__main__":
    for nombre, f in list(globals().items()):
        if nombre.startswith("test_"):
            f()
            print("ok", nombre)
