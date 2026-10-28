"""Inserta salida/cuadro.json en la plantilla y escribe salida/cuadro_mondragon.html.

Uso: python src/construir_dashboard.py [--demo]
"""
from __future__ import annotations

import json
import sys

from datos import RAIZ


def main(demo: bool = False) -> None:
    nombre = "cuadro_demo.json" if demo else "cuadro.json"
    datos = json.loads((RAIZ / "salida" / nombre).read_text(encoding="utf-8"))
    plantilla = (RAIZ / "dashboard" / "plantilla.html").read_text(encoding="utf-8")
    incrustado = json.dumps(datos, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = plantilla.replace("/*DATOS*/", incrustado)
    destino = RAIZ / "salida" / ("cuadro_mondragon_demo.html" if demo else "cuadro_mondragon.html")
    destino.write_text(html, encoding="utf-8")
    print(destino)


if __name__ == "__main__":
    main(demo="--demo" in sys.argv)
