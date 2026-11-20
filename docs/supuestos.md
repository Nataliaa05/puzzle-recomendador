# Supuestos declarados

Todo lo que **no** sale de datos reales se declara aquí, con su justificación y el notebook donde se usa.

| # | Supuesto | Por qué | Dónde |
|---|---|---|---|
| S1 | Menús, precios, ingredientes y alérgenos son sintéticos (biblioteca canónica) | No existe una fuente pública con menús | 02 |
| S2 | Presupuesto por visita = gasto trimestral por persona en alimentos fuera del hogar (ENIGH 2024) ÷ 8 salidas por trimestre | La ENIGH mide gasto del hogar por trimestre, no ticket por visita | 03 |
| S3 | La cocina de los restaurantes del DENUE se infiere del nombre + KNN | El SCIAN no distingue la mayoría de las cocinas | 01 |
| S4 | Los gustos latentes siguen las distribuciones de `config/simulation.yaml` | Es una verdad conocida, para poder evaluar | 03 |
| S5 | MNAR en calificaciones (califica quien quedó muy contento o muy molesto) | Sesgo documentado en sistemas de reseñas | 03 |
| S6 | La potencia estadística se calcula bajo el mundo simulado | No hay datos reales todavía | 13 |
| S7 | Valores iniciales de los parámetros = valores de trabajo, calibrados por hipercubo latino | Sin datos reales para estimarlos | 04 |
| S8 | El nivel de precios de un restaurante sube con la escolaridad promedio de su AGEB (`GRAPROES`, Censo 2020), con elasticidad configurable | Proxy socioeconómico real y público; no hay precios reales de menús | 02 |
| S10 | Prevalencia de alergias por alérgeno (`population.allergy_prevalence`), del orden de magnitud reportado para adultos | No hay registro poblacional de alergias en México | 03 |
| S11 | Los usuarios registrados salen de hogares con gasto en comida fuera > 0 (68 % de los hogares de la CDMX) | Quien usa Puzzle, por definición, come fuera | 03 |
| S12 | Agotados con tasa ∝ demanda latente del platillo en su restaurante (no de las visitas simuladas) | Los usuarios simulados son una fracción de la clientela real; su demanda sola no agotaría nada | 03 |
| S14 | Bandas plausibles de los hechos estilizados (`config/validation.yaml`), declaradas antes de ver los resultados | Conocimiento de dominio (regla 80/20, cola pesada de frecuencia); no hay datos reales para estimarlas | 04 |
| S15 | Perfil horario de apertura de mesas (`visits.hourly_profile`): picos de comida (14–15 h) y cena (20–21 h), horas de transición con peso bajo; una persona no se sienta en dos mesas a menos de 3 h (`min_hours_between_visits`) | No hay datos de horarios de consumo por hora en la CDMX; los picos siguen el horario de comida mexicano | 03 |
| S9 | Restaurantes sin palabra clave en el nombre ni señal para el KNN → cocina `mexicana` | El giro 722511 es "a la carta o comida corrida", mayoritariamente cocina mexicana | 01 |

**Advertencia general:** los datos sintéticos validan el pipeline, las métricas y la arquitectura, no el rendimiento en el mundo real. Ese se mide con los datos de Puzzle y con la prueba A/B.
