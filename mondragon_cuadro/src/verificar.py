"""Descarga todas las series y escribe salida/informe_series.md para revisarlas a mano
antes de estimar: etiqueta devuelta por la fuente frente a la ficha, cobertura y alertas.

Uso: python src/verificar.py
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from datos import RAIZ, catalogo, modelo
from descarga import SinDatos, descargar


def main() -> None:
    cat = catalogo()
    nucleo = set(modelo()["nucleo"]["variables"])
    hoy = pd.Period(date.today(), freq="M")
    filas = []
    for sid, ficha in cat.items():
        rol = "núcleo" if sid in nucleo else "satélite"
        try:
            m = descargar(sid, ficha)
        except SinDatos as e:
            filas.append(f"| {sid} | {rol} | {ficha['etiqueta']} | — | — | — | **NO DISPONIBLE**: {e} |")
            continue
        alertas = []
        if ficha["frecuencia"] == "M":
            ultimo = pd.Period(m["ultimo"], freq="M")
            if (hoy - ultimo).n > 4:
                alertas.append(f"desfasada {(hoy - ultimo).n} meses")
            if m["n"] < 96:
                alertas.append("menos de 8 años")
        else:
            ultimo = pd.Period(m["ultimo"].replace("-Q", "Q"), freq="Q")
            if (hoy.asfreq("Q") - ultimo).n > 2:
                alertas.append(f"desfasada {(hoy.asfreq('Q') - ultimo).n} trimestres")
            if m["n"] < 32:
                alertas.append("menos de 8 años")
        if rol == "núcleo" and m["primero"][:4] > "2002":
            alertas.append(f"empieza en {m['primero']}: acorta la muestra del núcleo")
        filas.append(f"| {sid} | {rol} | {ficha['etiqueta']} | {m['etiqueta_fuente'][:90]} | "
                     f"{m['primero']} → {m['ultimo']} | {m['ultimo_valor']:.3f} | {'; '.join(alertas) or 'ok'} |")
    salida = RAIZ / "salida"
    salida.mkdir(exist_ok=True)
    texto = ["# Informe de series", "",
             f"Generado el {date.today():%d-%m-%Y}. Comprobar que la etiqueta de la fuente mide lo que dice la ficha.", "",
             "| id | rol | ficha | etiqueta en la fuente | cobertura | último valor | alertas |",
             "|---|---|---|---|---|---|---|"] + filas
    (salida / "informe_series.md").write_text("\n".join(texto) + "\n", encoding="utf-8")
    print("\n".join(texto))


if __name__ == "__main__":
    main()
