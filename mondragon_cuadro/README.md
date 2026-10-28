# Cuadro macro MONDRAGON por división · escenario base

Proyección más probable para 2026-2029 de unas diez variables por división industrial
(Automoción, Máquina-herramienta y automatización, Componentes, Construcción) y del agregado
de Industria, con mediana y bandas. Resultado: `salida/cuadro_mondragon.html`.

## Límites

- Escenario base, sin riesgos concretos. Los escenarios estresados quedan fuera por ahora.
- Las variables sectoriales se estiman con datos públicos de cada rama (Eurostat, BCE, FRED,
  Eustat, Comext), no con datos de las cooperativas: no existen series públicas de ventas o
  EBITDA por cooperativa con historia suficiente. Las ventas de 2025 por división (Informe anual
  2025) solo se usan para ponderar el agregado de Industria.
- Los años 2026-2028 del núcleo se anclan al consenso publicado. Más allá, la senda converge a
  las medias de largo plazo de `config/modelo.yaml`, que son supuestos del equipo.
- Los códigos de las series se validan con `verificar.py` antes de estimar (paso 2); a 28-10-2026
  las 32 series descargan y están al día. Las series de la zona euro usan EA21 (con Bulgaria desde
  2026) y el IAPC, ECOICOP ver.2 (`prc_hicp_minr`). `pib_pv` se descarga de una consulta guardada
  de Eustat: si deja de responder, se vuelve a crear en la web de la tabla o se usa
  `datos/manuales/pib_pv.csv`.
- Las anclas del Banco de España proceden de prensa y deben cotejarse con el informe de
  proyecciones. Los supuestos técnicos del BCE (Euríbor, Brent, gas, tipo de cambio) proceden de la
  tabla A de las proyecciones de septiembre de 2026, con fecha de corte de mercados 19-ago-2026.

## Método

1. **Núcleo**: BVAR trimestral de 9 variables (PIB zona euro, España y País Vasco; IAPC España;
   Euríbor 3m; EUR/USD; Brent; gas; confianza industrial) con medias de largo plazo informativas
   (Villani, 2009), prior de Minnesota y dos retardos. Las ecuaciones de 2020T1-2021T2 se
   excluyen de la verosimilitud.
2. **Escenario más probable**: previsión condicionada (Waggoner y Zha, 1999). Condiciones:
   medias anuales del consenso con una tolerancia (blandas) y trimestres ya publicados de las
   series mensuales (duras; parciales si faltan meses). Un ancla que exige un salto mayor de
   3 desviaciones típicas frente a los datos ya publicados se descarta y se avisa.
3. **Satélites**: una ecuación bayesiana por variable sectorial, función del núcleo en t y t-1,
   con dummies trimestrales si la serie no está desestacionalizada. Se simulan sobre cada senda
   del núcleo, de modo que las bandas recogen la incertidumbre de ambos.
4. **Validación**: `backtest.py` compara el núcleo sin condicionar con un AR(1) a 1, 4 y 8
   trimestres desde 2012 y contrasta λ₁ ∈ {0,1; 0,2; 0,3}.

## Ejecución

```bash
pip install -r requirements.txt
python tests/test_motor.py                 # motor con datos simulados
python src/verificar.py                    # 1. descarga y salida/informe_series.md
# 2. revisar el informe: la etiqueta de la fuente debe medir lo que dice la ficha.
#    Corregir códigos en config/series.yaml; series sin API, en datos/manuales/<id>.csv (periodo,valor)
# 3. completar y cotejar las anclas de config/modelo.yaml
python src/backtest.py                     # 4. fijar lambda1 en config/modelo.yaml
python src/ejecutar.py                     # 5. salida/cuadro.json
python src/construir_dashboard.py          # 6. salida/cuadro_mondragon.html
```

Versión con datos simulados (forma del entregable, sin ninguna cifra real):
`python src/generar_demo.py && python src/backtest.py --demo --rapido && python src/ejecutar.py --demo && python src/construir_dashboard.py --demo`.

## Revisión antes de entregar

- `salida/informe_series.md`: ninguna serie desfasada ni con etiqueta distinta de la ficha.
- Avisos de `ejecutar.py`: anclas descartadas, satélites sin datos.
- R² de los satélites (panel «Calidad del modelo»): por debajo de 0,2 la variable apenas depende
  del núcleo; presentarla como inercia propia, no como resultado del escenario.
- 2026 debe reproducir los datos ya publicados; 2026-2027 del núcleo, el consenso dentro de su
  tolerancia.

## Estructura

```
config/series.yaml     catálogo de series: fuente, código, frecuencia, transformación
config/modelo.yaml     núcleo, priors, satélites, anclas y divisiones
src/bvar.py            BVAR de estado estacionario y previsión condicionada
src/satelites.py       ecuaciones satélite
src/cuadro.py          anclas anuales y resumen en el cuadro
src/descarga.py        descarga y caché en datos/brutos/
src/verificar.py       informe de series
src/backtest.py        evaluación fuera de muestra
src/ejecutar.py        escenario base completo
src/construir_dashboard.py
dashboard/plantilla.html
```
