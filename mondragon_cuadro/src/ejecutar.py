"""Escenario base: estima el núcleo, lo condiciona a las anclas y a los datos ya
publicados del trimestre en curso, simula los satélites y escribe salida/cuadro.json.

Uso: python src/ejecutar.py [--demo]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from bvar import PriorBVAR, estimar, prever
from cuadro import condicion_anual, condicion_trimestre, resumen
from datos import BRUTOS, RAIZ, a_trimestral, catalogo, leer_bruta, modelo, nivel_para_salida, transformar
from satelites import estimar_satelite, simular_satelite


def cargar(cat: dict, carpeta) -> tuple[dict, dict, dict, list]:
    trimestrales, parciales, metas, avisos = {}, {}, {}, []
    for sid, ficha in cat.items():
        s = leer_bruta(sid, carpeta)
        if s is None or len(s) == 0:
            avisos.append(f"{sid}: sin datos en caché; queda fuera del cuadro")
            continue
        q, parcial = a_trimestral(s, ficha)
        trimestrales[sid] = q
        if parcial:
            parciales[sid] = parcial
        meta = carpeta / f"{sid}.json"
        metas[sid] = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    return trimestrales, parciales, metas, avisos


def main(demo: bool = False) -> dict:
    cat, mod = catalogo(), modelo()
    carpeta = RAIZ / "datos" / ("demo" if demo else "brutos")
    q, parciales, metas, avisos = cargar(cat, carpeta)

    cfg_n = mod["nucleo"]
    nucleo = [v for v in cfg_n["variables"] if v in q]
    for v in cfg_n["variables"]:
        if v not in q:
            avisos.append(f"núcleo sin {v}: se estima sin esa variable")
    inicio = pd.Period(mod["muestra"]["inicio"], freq="Q")
    fin = pd.Period(mod["muestra"]["horizonte_fin"], freq="Q")

    trans = {v: transformar(q[v], cat[v]) for v in q}
    panel = pd.DataFrame({v: trans[v] for v in nucleo})
    panel = panel[panel.index >= inicio]
    completo = panel.dropna()
    primero = completo.index[0]
    panel = panel[panel.index >= primero]
    T_fin = panel.dropna().index[-1]
    hist = panel.loc[:T_fin]
    if hist.isna().any().any():
        raise RuntimeError("huecos dentro de la muestra del núcleo: " + str(hist.columns[hist.isna().any()].tolist()))
    y = hist.values
    trims_hist = hist.index
    H = (fin - T_fin).n
    trims_fut = pd.period_range(T_fin + 1, periods=H, freq="Q")

    excl = np.isin(trims_hist.astype(str), mod["muestra"]["excluir"])
    medias, sds, propios = [], [], []
    for v in nucleo:
        c = cfg_n["variables"][v]
        medias.append(float(hist[v].mean()) if c["media"] == "muestral" else float(c["media"]))
        sds.append(float(c["sd"]))
        propios.append(float(c["propio"]))
    prior = PriorBVAR(media_mu=np.array(medias), sd_mu=np.array(sds), media_propio_retardo=np.array(propios),
                      lambda1=cfg_n["lambda1"], lambda2=cfg_n["lambda2"])
    post = estimar(y, nucleo, p=cfg_n["retardos"], prior=prior, excluir=excl,
                   n_draws=cfg_n["sorteos"], burn=cfg_n["descarte"], semilla=11)

    condiciones, anclas_usadas = [], []
    for a in ([] if demo else mod.get("anclas", [])):
        v = a["variable"]
        if v not in nucleo or a.get("valor") is None:
            continue
        i = nucleo.index(v)
        tipo = "crecimiento" if cat[v]["transformacion"] == "dlog" else "nivel"
        c = condicion_anual(i, tipo, int(a["anio"]), float(a["valor"]), float(a["sd"]),
                            trims_hist, y, H, etiqueta=f"{v} {a['anio']}")
        if c is None:
            avisos.append(f"ancla {v} {a['anio']} fuera del horizonte o ya observada; no se aplica")
            continue
        peso_libre = sum(c.pesos.values())
        exigido = c.valor / peso_libre
        if tipo == "crecimiento":
            z = (exigido - prior.media_mu[i]) / hist[v].std()
        else:
            z = (exigido - hist[v].iloc[-1]) / hist[v].std()
        if abs(z) > 3:
            avisos.append(f"ancla {v} {a['anio']} incompatible con los datos ya publicados "
                          f"(exige {exigido:.2f} por trimestre libre, z={z:.1f}); no se aplica")
            continue
        condiciones.append(c)
        anclas_usadas.append({**a, "etiqueta": cat[v]["etiqueta"], "z_exigencia": round(float(z), 2)})

    datos_recientes = []
    for v in nucleo:
        s = trans[v]
        for t in s.index[s.index > T_fin]:
            if pd.isna(s.loc[t]):
                continue
            h = (t - T_fin).n - 1
            condiciones.append(condicion_trimestre(nucleo.index(v), h, float(s.loc[t]), 0.0, f"{v} {t}"))
            datos_recientes.append({"variable": v, "trimestre": str(t), "tipo": "observado"})
        if v in parciales and cat[v].get("ajuste") != "NSA_DESEST":
            p = parciales[v]
            t = pd.Period(p["trimestre"], freq="Q")
            if t <= T_fin or (t - T_fin).n > H:
                continue
            h = (t - T_fin).n - 1
            previo = q[v].get(t - 1)
            if previo is None or pd.isna(previo):
                continue
            valor = 100 * np.log(p["valor"] / previo) if cat[v]["transformacion"] == "dlog" else p["valor"]
            sd = float(hist[v].diff().std() if cat[v]["transformacion"] == "nivel" else hist[v].std()) * (3 - p["meses"]) / 3
            condiciones.append(condicion_trimestre(nucleo.index(v), h, float(valor), sd, f"{v} {t} parcial"))
            datos_recientes.append({"variable": v, "trimestre": str(t), "tipo": f"parcial ({p['meses']} meses)"})

    sendas = prever(post, y, H, condiciones, semilla=12)
    D = sendas.shape[0]
    anios = list(range(T_fin.year, fin.year + 1))

    resultados, fichas = {}, {}
    for i, v in enumerate(nucleo):
        tipo = "dlog" if cat[v]["transformacion"] == "dlog" else "nivel"
        nivel = nivel_para_salida(q[v], cat[v]).loc[:T_fin] if tipo == "dlog" else None
        resultados[v] = resumen(v, tipo, hist[v], sendas[:, :, i], trims_fut, anios, nivel_hist=nivel)

    necesarios = {x for d in mod["divisiones"].values() for x in d["variables"]}
    necesarios |= set(mod["divisiones"]["industria"].get("ponderacion", {}))
    X_hist = hist.values
    diagnostico_sat, sim_sat = [], {}
    for sid, cfg in mod["satelites"].items():
        if sid not in necesarios:
            continue
        if sid not in q:
            continue
        faltan = [k for k in cfg["impulsores"] if k not in nucleo]
        if faltan:
            avisos.append(f"{sid}: impulsores ausentes del núcleo {faltan}; queda fuera")
            continue
        s_tr = trans[sid]
        s_al = s_tr.reindex(trims_hist).values
        try:
            sat = estimar_satelite(sid, s_al, X_hist, [nucleo.index(k) for k in cfg["impulsores"]],
                                   trims_hist.quarter.values, estacional=cat[sid]["ajuste"] == "NSA",
                                   rho0=cfg["rho0"], excluir=excl, n_draws=D, semilla=21)
        except ValueError as e:
            avisos.append(str(e))
            continue
        ult = s_tr.dropna().index[-1]
        if ult < trims_hist[0]:
            avisos.append(f"{sid}: sin datos en la muestra del núcleo")
            continue
        X_full = np.concatenate([np.repeat(X_hist[None], D, axis=0), sendas], axis=1)
        trims_full = trims_hist.append(trims_fut)
        pos = trims_full.get_loc(ult)
        X_sim = X_full[:, pos:, :]
        fut_s = trims_full[pos + 1:]
        sim = simular_satelite(sat, float(s_tr.loc[ult]), X_sim, fut_s.quarter.values, semilla=31)
        tipo = "dlog" if cat[sid]["transformacion"] == "dlog" else "nivel"
        nivel = nivel_para_salida(q[sid], cat[sid]).loc[:ult] if tipo == "dlog" else None
        resultados[sid] = resumen(sid, tipo, s_tr.loc[:ult], sim, fut_s, anios, nivel_hist=nivel)
        sim_sat[sid] = (fut_s, sim, nivel)
        diagnostico_sat.append({"id": sid, "n_obs": sat.n_obs, "r2": round(sat.r2, 3),
                                "impulsores": cfg["impulsores"], "ultimo_dato": str(ult)})

    pond = mod["divisiones"]["industria"].get("ponderacion", {})
    disponibles = {k: w for k, w in pond.items() if k in sim_sat}
    if len(disponibles) >= 2:
        wtot = sum(disponibles.values())
        inicio_f = min(sim_sat[k][0][0] for k in disponibles)
        fut_c = pd.period_range(inicio_f, fin, freq="Q")
        dl_fut = np.zeros((D, len(fut_c)))
        partes_hist = []
        for k, w in disponibles.items():
            fut_k, sim_k, nivel_k = sim_sat[k]
            serie_k = 100 * np.log(nivel_k).diff()
            futuro_df = pd.DataFrame(sim_k.T, index=fut_k)
            combinado = pd.concat([pd.DataFrame(np.repeat(serie_k.values[:, None], D, axis=1), index=serie_k.index),
                                   futuro_df])
            dl_fut += (w / wtot) * combinado.reindex(fut_c).values.T
            partes_hist.append(serie_k.loc[:inicio_f - 1] * (w / wtot))
        dl_hist = pd.concat(partes_hist, axis=1).dropna().sum(axis=1)
        dl_hist = dl_hist.loc[:inicio_f - 1]
        if len(dl_hist) == 0 or dl_hist.index[-1] != inicio_f - 1:
            avisos.append("actividad ponderada: historia incompleta antes de la proyección")
        nivel_pond = 100 * np.exp(dl_hist.cumsum() / 100)
        resultados["actividad_ponderada"] = resumen("actividad_ponderada", "dlog", dl_hist, dl_fut, fut_c, anios,
                                                    nivel_hist=nivel_pond)
        fichas["actividad_ponderada"] = {
            "etiqueta": "Actividad industrial ponderada por ventas de las divisiones",
            "unidad": "% var. media anual", "fuente": "IPI C29, C28, C27 y construcción (Eurostat); pesos: ventas 2025",
            "rol": "satélite", "tipo": "dlog"}

    for sid in resultados:
        if sid in cat:
            fichas[sid] = {"etiqueta": cat[sid]["etiqueta"], "unidad": cat[sid]["unidad"],
                           "fuente": f"{cat[sid]['fuente']}: {cat[sid].get('dataset') or cat[sid].get('serie') or cat[sid].get('clave') or cat[sid].get('tabla')}",
                           "rol": "núcleo" if sid in nucleo else "satélite",
                           "tipo": "dlog" if cat[sid]["transformacion"] == "dlog" else "nivel",
                           "ultimo_dato": metas.get(sid, {}).get("ultimo"),
                           "anclada": any(a["variable"] == sid for a in anclas_usadas)}

    divisiones = {}
    for k, d in mod["divisiones"].items():
        divisiones[k] = {"nombre": d["nombre"], "ventas_2025": d.get("ventas_2025"),
                         "fuente_ventas": d.get("fuente_ventas"), "cooperativas": d.get("cooperativas", ""),
                         "variables": [v for v in d["variables"] if v in resultados],
                         "ausentes": [v for v in d["variables"] if v not in resultados]}

    salida = {
        "meta": {"generado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "simulado": demo, "ultimo_trimestre_nucleo": str(T_fin),
                 "muestra": f"{trims_hist[0]}–{T_fin}", "excluidos": mod["muestra"]["excluir"],
                 "sorteos": int(D), "retardos": post.p, "lambda1": prior.lambda1,
                 "descartes_explosivos": post.descartes_explosivos, "anios": anios},
        "anclas": anclas_usadas, "datos_recientes": datos_recientes,
        "estado_estacionario": {v: {"media_prior": medias[i], "media_posterior": round(float(post.mu[:, i].mean()), 3),
                                    "nota": cfg_n["variables"][v].get("nota", "")} for i, v in enumerate(nucleo)},
        "satelites": diagnostico_sat, "avisos": avisos,
        "fichas": fichas, "divisiones": divisiones, "resultados": resultados,
    }
    bt = RAIZ / "salida" / ("backtest_demo.json" if demo else "backtest.json")
    if bt.exists():
        salida["backtest"] = json.loads(bt.read_text(encoding="utf-8"))
    (RAIZ / "salida").mkdir(exist_ok=True)
    nombre = "cuadro_demo.json" if demo else "cuadro.json"
    (RAIZ / "salida" / nombre).write_text(json.dumps(salida, ensure_ascii=False), encoding="utf-8")
    print(f"Núcleo {trims_hist[0]}–{T_fin}, {len(nucleo)} variables, {D} sorteos, "
          f"{len(condiciones)} condiciones, {len(diagnostico_sat)} satélites. Avisos: {len(avisos)}")
    for a in avisos:
        print("  -", a)
    return salida


if __name__ == "__main__":
    main(demo="--demo" in sys.argv)
