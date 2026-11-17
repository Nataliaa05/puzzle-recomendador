# Supuestos declarados

Todo lo que **no** sale de datos reales se declara aquí, con su justificación y el notebook donde se usa.

| # | Supuesto | Por qué | Dónde |
|---|---|---|---|
| S1 | Menús, precios, ingredientes y alérgenos son sintéticos (biblioteca canónica) | No existe una fuente pública con menús | 02 |
| S2 | El ticket por visita se deriva del gasto trimestral del hogar de la ENIGH, suponiendo un tamaño de hogar y una frecuencia de salidas | La ENIGH mide gasto del hogar por trimestre, no ticket por persona | 03 |
| S3 | La cocina de los restaurantes del DENUE se infiere del nombre + KNN | El SCIAN no distingue la mayoría de las cocinas | 01 |
| S4 | Los gustos latentes siguen las distribuciones de `config/simulation.yaml` | Es una verdad conocida, para poder evaluar | 03 |
| S5 | MNAR en calificaciones (califica quien quedó muy contento o muy molesto) | Sesgo documentado en sistemas de reseñas | 03 |
| S6 | La potencia estadística se calcula bajo el mundo simulado | No hay datos reales todavía | 13 |
| S7 | Valores iniciales de los parámetros = valores de trabajo, calibrados por hipercubo latino | Sin datos reales para estimarlos | 04 |

**Advertencia general:** los datos sintéticos validan el pipeline, las métricas y la arquitectura, no el rendimiento en el mundo real. Ese se mide con los datos de Puzzle y con la prueba A/B.
