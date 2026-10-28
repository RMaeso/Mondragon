"""BVAR de estado estacionario (Villani, 2009) con previsión condicionada.

Modelo: A(L) (y_t - mu) = e_t,  e_t ~ N(0, S).
Priors: Minnesota independiente para los coeficientes, normal para mu,
Jeffreys para S. Muestreo de Gibbs en tres bloques: B | S, mu; S | B, mu; mu | B, S.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg


@dataclass
class PriorBVAR:
    media_mu: np.ndarray
    sd_mu: np.ndarray
    media_propio_retardo: np.ndarray
    lambda1: float = 0.2
    lambda2: float = 0.5
    decaimiento: float = 2.0


@dataclass
class Posterior:
    nombres: list[str]
    p: int
    B: np.ndarray  # (draws, n*p, n)
    S: np.ndarray  # (draws, n, n)
    mu: np.ndarray  # (draws, n)
    descartes_explosivos: int = 0
    info: dict = field(default_factory=dict)


def _retardos(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    T = y.shape[0]
    Y = y[p:]
    X = np.hstack([y[p - l: T - l] for l in range(1, p + 1)])
    return Y, X


def _sd_ar(y: np.ndarray, p: int) -> np.ndarray:
    sd = np.empty(y.shape[1])
    for i in range(y.shape[1]):
        Yi, Xi = _retardos(y[:, [i]], p)
        Xi = np.hstack([np.ones((Xi.shape[0], 1)), Xi])
        coef, *_ = np.linalg.lstsq(Xi, Yi, rcond=None)
        res = Yi - Xi @ coef
        sd[i] = np.sqrt(res.var(ddof=Xi.shape[1]))
    return sd


def _companion_estable(B: np.ndarray, n: int, p: int) -> bool:
    C = np.zeros((n * p, n * p))
    C[:n, :] = B.T
    if p > 1:
        C[n:, :-n] = np.eye(n * (p - 1))
    return np.max(np.abs(np.linalg.eigvals(C))) < 0.999


def estimar(y: np.ndarray, nombres: list[str], p: int, prior: PriorBVAR,
            excluir: np.ndarray | None = None, n_draws: int = 2000, burn: int = 500,
            semilla: int = 0) -> Posterior:
    """Estima el BVAR. `excluir` marca con True las filas de y cuyas ecuaciones
    se eliminan de la verosimilitud (p. ej., 2020T1-2021T2)."""
    rng = np.random.default_rng(semilla)
    T, n = y.shape
    k = n * p
    sd = _sd_ar(y, p)

    b0 = np.zeros((k, n))
    for i in range(n):
        b0[i, i] = prior.media_propio_retardo[i]
    var_b = np.empty((k, n))
    for l in range(1, p + 1):
        for j in range(n):
            for i in range(n):
                fila = (l - 1) * n + j
                escala = prior.lambda1 / l ** prior.decaimiento
                if i != j:
                    escala *= prior.lambda2 * sd[i] / sd[j]
                var_b[fila, i] = escala ** 2
    prec_b = np.diag(1.0 / var_b.flatten(order="F"))
    b0_vec = b0.flatten(order="F")

    filas = np.arange(p, T)
    if excluir is not None:
        filas = filas[~excluir[p:]]

    mu = prior.media_mu.copy()
    S = np.diag(sd ** 2)
    prec_mu0 = np.diag(1.0 / prior.sd_mu ** 2)

    B_draws, S_draws, mu_draws = [], [], []
    descartes = 0
    total = n_draws + burn
    it = 0
    intentos = 0
    B = b0.copy()
    while it < total:
        intentos += 1
        if intentos > total * 20:
            raise RuntimeError("Demasiados sorteos explosivos: revisar especificación")
        x = y - mu
        Yx, Xx = _retardos(x, p)
        Yx = Yx[filas - p]
        Xx = Xx[filas - p]
        Si = np.linalg.inv(S)
        prec_post = prec_b + np.kron(Si, Xx.T @ Xx)
        rhs = prec_b @ b0_vec + (Xx.T @ Yx @ Si).flatten(order="F")
        L = np.linalg.cholesky(prec_post)
        media = linalg.cho_solve((L, True), rhs)
        b = media + linalg.solve_triangular(L.T, rng.standard_normal(k * n), lower=False)
        B_nuevo = b.reshape((k, n), order="F")
        if not _companion_estable(B_nuevo, n, p):
            descartes += 1
            continue
        B = B_nuevo

        E = Yx - Xx @ B
        S = linalg.inv(_wishart(rng, len(filas), np.linalg.inv(E.T @ E)))

        suma_B = sum(B[(l - 1) * n: l * n, :].T for l in range(1, p + 1))
        U = np.eye(n) - suma_B
        z = []
        for t in filas:
            zt = y[t].copy()
            for l in range(1, p + 1):
                zt -= B[(l - 1) * n: l * n, :].T @ y[t - l]
            z.append(zt)
        z = np.array(z)
        Si = np.linalg.inv(S)
        prec_mu = prec_mu0 + len(filas) * U.T @ Si @ U
        rhs_mu = prec_mu0 @ prior.media_mu + U.T @ Si @ z.sum(axis=0)
        Lm = np.linalg.cholesky(prec_mu)
        media_mu = linalg.cho_solve((Lm, True), rhs_mu)
        mu = media_mu + linalg.solve_triangular(Lm.T, rng.standard_normal(n), lower=False)

        if it >= burn:
            B_draws.append(B.copy())
            S_draws.append(S.copy())
            mu_draws.append(mu.copy())
        it += 1

    return Posterior(nombres=list(nombres), p=p, B=np.array(B_draws), S=np.array(S_draws),
                     mu=np.array(mu_draws), descartes_explosivos=descartes,
                     info={"T_efectivo": int(len(filas)), "lambda1": prior.lambda1,
                           "lambda2": prior.lambda2, "sd_ar": sd.tolist()})


def _wishart(rng: np.random.Generator, gl: int, escala: np.ndarray) -> np.ndarray:
    n = escala.shape[0]
    L = np.linalg.cholesky(escala)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = np.sqrt(rng.chisquare(gl - i))
        A[i, :i] = rng.standard_normal(i)
    LA = L @ A
    return LA @ LA.T


def respuestas(B: np.ndarray, n: int, p: int, H: int) -> np.ndarray:
    """Psi[h] (n x n): efecto en y_{T+1+h} de una innovación en T+1."""
    Psi = np.zeros((H, n, n))
    Psi[0] = np.eye(n)
    for h in range(1, H):
        acc = np.zeros((n, n))
        for l in range(1, min(h, p) + 1):
            acc += B[(l - 1) * n: l * n, :].T @ Psi[h - l]
        Psi[h] = acc
    return Psi


def senda_media(B: np.ndarray, mu: np.ndarray, historia: np.ndarray, p: int, H: int) -> np.ndarray:
    n = historia.shape[1]
    x = list(historia[-p:] - mu)
    out = np.zeros((H, n))
    for h in range(H):
        xt = np.zeros(n)
        for l in range(1, p + 1):
            xt += B[(l - 1) * n: l * n, :].T @ x[-l]
        x.append(xt)
        out[h] = xt + mu
    return out


@dataclass
class Condicion:
    """Restricción lineal sobre la senda futura: sum(pesos * y[h, i]) = valor (± sd)."""
    pesos: dict[tuple[int, int], float]
    valor: float
    sd: float = 0.0
    etiqueta: str = ""


def prever(post: Posterior, historia: np.ndarray, H: int, condiciones: list[Condicion] | None = None,
           semilla: int = 1) -> np.ndarray:
    """Sendas futuras (draws, H, n). Con condiciones, las innovaciones se sortean de su
    distribución condicionada a que la senda cumpla las restricciones (Waggoner y Zha, 1999);
    las condiciones con sd > 0 se tratan como blandas sorteando primero su valor."""
    rng = np.random.default_rng(semilla)
    D, n = post.mu.shape
    p = post.p
    condiciones = condiciones or []
    out = np.empty((D, H, n))
    for d in range(D):
        B, S, mu = post.B[d], post.S[d], post.mu[d]
        m = senda_media(B, mu, historia, p, H)
        Psi = respuestas(B, n, p, H)
        Lc = np.linalg.cholesky(S)
        e0 = (Lc @ rng.standard_normal((n, H))).T
        if not condiciones:
            out[d] = m + _propagar(Psi, e0)
            continue
        q = len(condiciones)
        G = np.zeros((q, H * n))
        r = np.empty(q)
        base = m + _propagar(Psi, e0)
        for c_i, c in enumerate(condiciones):
            objetivo = c.valor + (c.sd * rng.standard_normal() if c.sd > 0 else 0.0)
            actual = 0.0
            for (h, i), w in c.pesos.items():
                actual += w * base[h, i]
                for s in range(h + 1):
                    G[c_i, s * n:(s + 1) * n] += w * Psi[h - s][i, :]
            r[c_i] = objetivo - actual
        Om = np.kron(np.eye(H), S)
        GOG = G @ Om @ G.T
        ajuste = Om @ G.T @ np.linalg.pinv(GOG) @ r
        e = e0 + ajuste.reshape(H, n)
        out[d] = m + _propagar(Psi, e)
    return out


def _propagar(Psi: np.ndarray, e: np.ndarray) -> np.ndarray:
    H, n = e.shape
    y = np.zeros((H, n))
    for h in range(H):
        for s in range(h + 1):
            y[h] += Psi[h - s] @ e[s]
    return y
