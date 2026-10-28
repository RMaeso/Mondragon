"""Ecuaciones satélite: variables sectoriales que dependen del núcleo sin realimentarlo.

s_t = c + rho * s_{t-1} + sum_k (b_k0 x_{k,t} + b_k1 x_{k,t-1}) + estacionalidad + e_t

Regresión bayesiana normal-gamma inversa con prior de contracción hacia cero
(salvo rho, centrado en `rho0`). Se simula a lo largo de cada senda del núcleo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Satelite:
    nombre: str
    impulsores: list[int]
    estacional: bool
    coef: np.ndarray        # (draws, k)
    sigma: np.ndarray       # (draws,)
    columnas: list[str]
    n_obs: int
    r2: float


def _diseno(s: np.ndarray, X: np.ndarray, impulsores: list[int], estacional: bool,
            trimestre: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    T = len(s)
    cols = [np.ones(T - 1), s[:-1]]
    nombres = ["const", "rho"]
    for k in impulsores:
        cols += [X[1:, k], X[:-1, k]]
        nombres += [f"x{k}_0", f"x{k}_1"]
    if estacional:
        for q in (2, 3, 4):
            cols.append((trimestre[1:] == q).astype(float))
            nombres.append(f"q{q}")
    Z = np.column_stack(cols)
    return Z, s[1:], nombres


def estimar_satelite(nombre: str, s: np.ndarray, X: np.ndarray, impulsores: list[int],
                     trimestre: np.ndarray, estacional: bool = False, rho0: float = 0.3,
                     escala_prior: float = 1.0, n_draws: int = 2000, semilla: int = 0,
                     excluir: np.ndarray | None = None) -> Satelite:
    rng = np.random.default_rng(semilla)
    valido = ~np.isnan(s) & ~np.isnan(X[:, impulsores]).any(axis=1)
    Z, yv, nombres = _diseno(s, X, impulsores, estacional, trimestre)
    filas = valido[1:] & valido[:-1]
    if excluir is not None:
        filas &= ~excluir[1:]
    Z, yv = Z[filas], yv[filas]
    if len(yv) < 20:
        raise ValueError(f"{nombre}: solo {len(yv)} observaciones válidas")

    k = Z.shape[1]
    sd_y = np.std(yv)
    sd_x = np.where(np.std(Z, axis=0) > 0, np.std(Z, axis=0), 1.0)
    b0 = np.zeros(k)
    b0[1] = rho0
    v0 = (escala_prior * sd_y / sd_x) ** 2
    v0[0] = (10 * sd_y) ** 2
    v0[1] = 0.25 ** 2
    P0 = np.diag(1 / v0)
    Pn = P0 + Z.T @ Z
    Vn = np.linalg.inv(Pn)
    bn = Vn @ (P0 @ b0 + Z.T @ yv)
    a_n = 2 + len(yv) / 2
    res = yv - Z @ bn
    b_n = 0.5 * (res @ res + (bn - b0) @ P0 @ (bn - b0)) + 0.5 * sd_y ** 2
    sig2 = b_n / rng.gamma(a_n, 1.0, size=n_draws)
    L = np.linalg.cholesky(Vn)
    coef = bn + (np.sqrt(sig2)[:, None] * (rng.standard_normal((n_draws, k)) @ L.T))
    coef[:, 1] = np.clip(coef[:, 1], -0.95, 0.95)
    r2 = 1 - (res @ res) / ((yv - yv.mean()) @ (yv - yv.mean()))
    return Satelite(nombre, impulsores, estacional, coef, np.sqrt(sig2), nombres, int(len(yv)), float(r2))


def simular_satelite(sat: Satelite, s_ult: float, X: np.ndarray, trimestres: np.ndarray,
                     semilla: int = 0) -> np.ndarray:
    """Simula desde el último dato observado del satélite.

    X: (draws, H+1, n) con el núcleo en el periodo del último dato (fila 0) y en los H
    periodos siguientes; `trimestres` (H,) son los trimestres del año de esos H periodos.
    Devuelve (draws, H).
    """
    rng = np.random.default_rng(semilla)
    D, H1, _ = X.shape
    H = H1 - 1
    idx = rng.integers(0, len(sat.sigma), size=D)
    C = sat.coef[idx]
    sig = sat.sigma[idx]
    out = np.empty((D, H))
    prev = np.full(D, s_ult)
    for h in range(H):
        v = C[:, 0] + C[:, 1] * prev
        j = 2
        for kk in sat.impulsores:
            v += C[:, j] * X[:, h + 1, kk] + C[:, j + 1] * X[:, h, kk]
            j += 2
        if sat.estacional:
            for qq in (2, 3, 4):
                if trimestres[h] == qq:
                    v += C[:, j]
                j += 1
        v += sig * rng.standard_normal(D)
        out[:, h] = v
        prev = v
    return out
