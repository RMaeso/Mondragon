"""Descarga de series a datos/brutos/<id>.csv (periodo, valor) con metadatos en <id>.json.

Fuentes: Eurostat y Comext (JSON-stat), BCE (SDMX csv), FRED (csv), Eustat (consulta guardada PX-Web)
y ficheros manuales en datos/manuales/<id>.csv con columnas periodo, valor.

Uso: python src/descarga.py [id ...]
"""
from __future__ import annotations

import io
import json
import sys
import time
from datetime import datetime, timezone
from itertools import product as producto

import numpy as np
import pandas as pd
import requests

from datos import BRUTOS, RAIZ, catalogo, normalizar_periodo

MANUALES = RAIZ / "datos" / "manuales"
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
COMEXT = "https://ec.europa.eu/eurostat/api/comext/dissemination/statistics/1.0/data/"
BCE = "https://data-api.ecb.europa.eu/service/data/"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="


class SinDatos(Exception):
    pass


def _get(url: str, **kw) -> requests.Response:
    for intento in range(3):
        try:
            r = requests.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
            if r.status_code in (400, 404):
                raise SinDatos(f"HTTP {r.status_code}: {r.text[:200]}")
        except requests.RequestException as e:
            if intento == 2:
                raise SinDatos(str(e)) from e
        time.sleep(2 * (intento + 1))
    raise SinDatos(f"sin respuesta válida: {url}")


def _jsonstat(js: dict) -> tuple[pd.Series, str]:
    ids = js["id"]
    tam = js["size"]
    dims = js["dimension"]
    t_pos = ids.index("time")
    cat_t = dims["time"]["category"]["index"]
    tiempos = sorted(cat_t, key=cat_t.get)
    pasos = np.cumprod([1] + tam[::-1][:-1])[::-1]
    valores = {int(k): v for k, v in js.get("value", {}).items()}
    serie = {}
    rangos = [range(n) for n in tam]
    for combo in producto(*rangos):
        flat = int(sum(c * s for c, s in zip(combo, pasos)))
        if flat in valores and valores[flat] is not None:
            t = tiempos[combo[t_pos]]
            serie[t] = serie.get(t, 0.0) + float(valores[flat])
    etiqueta = js.get("label", "")
    return pd.Series(serie).sort_index(), etiqueta


def _eurostat(base: str, ficha: dict) -> tuple[pd.Series, str, str]:
    juegos = [ficha["filtros"]] + ficha.get("alternativas", [])
    ultimo_error = None
    for filtros in juegos:
        params = [("format", "JSON"), ("lang", "EN")]
        for k, v in filtros.items():
            for vv in (v if isinstance(v, list) else [v]):
                params.append((k, vv))
        url = base + ficha["dataset"] + "?" + "&".join(f"{k}={v}" for k, v in params)
        try:
            js = _get(url).json()
            s, etq = _jsonstat(js)
            if len(s):
                return s, etq, url
            ultimo_error = "respuesta vacía"
        except SinDatos as e:
            ultimo_error = str(e)
    raise SinDatos(ultimo_error)


def _bce(ficha: dict) -> tuple[pd.Series, str, str]:
    url = f"{BCE}{ficha['flujo']}/{ficha['clave']}?format=csvdata"
    df = pd.read_csv(io.StringIO(_get(url).text))
    etq = str(df["TITLE"].iloc[0]) if "TITLE" in df.columns else ficha["clave"]
    return pd.Series(df["OBS_VALUE"].values, index=df["TIME_PERIOD"].astype(str).values), etq, url


def _fred(ficha: dict) -> tuple[pd.Series, str, str]:
    url = FRED + ficha["serie"]
    df = pd.read_csv(io.StringIO(_get(url).text))
    col_fecha = df.columns[0]
    v = pd.to_numeric(df[ficha["serie"]], errors="coerce")
    return pd.Series(v.values, index=df[col_fecha].astype(str).values).dropna(), ficha["serie"], url


def _eustat(ficha: dict) -> tuple[pd.Series, str, str]:
    """Consulta guardada de Eustat en JSON-stat 2 con una sola serie (todas las dimensiones
    salvo el periodo fijadas a un valor). La API PX-Web de Eustat no admite consultas, y la
    consulta guardada solo existe en uno de sus servidores: se reintenta hasta dar con él."""
    url = ficha.get("consulta_guardada")
    if not url:
        raise SinDatos("consulta guardada de Eustat sin configurar en config/series.yaml")
    js = None
    for intento in range(8):
        try:
            r = requests.get(url, timeout=60)
        except requests.RequestException:
            r = None
        if r is not None and r.status_code == 200 and r.headers.get("Content-Type", "").find("json") >= 0:
            js = r.json()
            break
        time.sleep(1)
    if js is None:
        raise SinDatos(f"Eustat: la consulta guardada no responde ({url})")
    if [n for d, n in zip(js["id"], js["size"]) if n > 1 and d != "periodo"]:
        raise SinDatos("Eustat: la consulta guardada devuelve más de una serie")
    cat = js["dimension"]["periodo"]["category"]
    codigos = sorted(cat["index"], key=cat["index"].get)
    # Los códigos son del tipo 20262 y las etiquetas 2026-2; se pasan a 2026Q2.
    periodos = [cat.get("label", {}).get(c, c).replace("-", "Q") for c in codigos]
    s = pd.Series(js["value"], index=periodos, dtype=float).dropna()
    return s, js.get("label", ficha["tabla"]), url


def _manual(sid: str) -> tuple[pd.Series, str, str] | None:
    f = MANUALES / f"{sid}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, dtype={"periodo": str})
    return pd.Series(df["valor"].values, index=df["periodo"].values), f"manual: {f.name}", str(f)


def descargar(sid: str, ficha: dict) -> dict:
    man = _manual(sid)
    if man is not None:
        s, etq, origen = man
    else:
        fuente = ficha["fuente"]
        if fuente == "eurostat":
            s, etq, origen = _eurostat(EUROSTAT, ficha)
        elif fuente == "comext":
            if ficha.get("dataset", "POR_CONFIRMAR") == "POR_CONFIRMAR":
                raise SinDatos("conjunto Comext por confirmar; alternativa: CSV de Datacomex en datos/manuales/")
            s, etq, origen = _eurostat(COMEXT, ficha)
        elif fuente == "bce":
            s, etq, origen = _bce(ficha)
        elif fuente == "fred":
            s, etq, origen = _fred(ficha)
        elif fuente == "eustat":
            s, etq, origen = _eustat(ficha)
        else:
            raise SinDatos(f"fuente desconocida {fuente}")
    s.index = [normalizar_periodo(p, ficha["frecuencia"]) for p in s.index]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    BRUTOS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"periodo": s.index, "valor": s.values}).to_csv(BRUTOS / f"{sid}.csv", index=False)
    meta = {"id": sid, "etiqueta_fuente": etq, "origen": origen, "n": int(len(s)),
            "primero": s.index[0], "ultimo": s.index[-1], "ultimo_valor": float(s.iloc[-1]),
            "descargado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    (BRUTOS / f"{sid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def main(ids: list[str]) -> int:
    cat = catalogo()
    ids = ids or list(cat)
    fallos = 0
    for sid in ids:
        try:
            m = descargar(sid, cat[sid])
            print(f"OK    {sid:32s} {m['primero']} → {m['ultimo']}  n={m['n']}")
        except SinDatos as e:
            fallos += 1
            print(f"FALLO {sid:32s} {e}")
    return fallos


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1:]) else 0)
