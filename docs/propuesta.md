# Recomendador de platillos de Puzzle: propuesta técnica

**Autora:** Natalia
**Fecha:** 2026-11-16
**Doble propósito:**

- **Proyecto del curso:** ciclo CRISP-DM completo, con datos sintéticos.
- **Producto Puzzle:** el mismo código pasa a producción cuando lleguen los datos reales.

## 1. Problema y objetivo

Un comensal registrado entra a un restaurante, muchas veces **por primera vez**, y Puzzle le muestra dos listas ordenadas según sus gustos:

- **"Para ti":** platillos individuales, por tipo (entradas, fuertes, bebidas, postres).
- **"Para la mesa":** platillos para compartir, según los gustos combinados de la mesa.

**La idea clave:** en un restaurante nuevo, el comensal nunca interactuó con sus platillos. Lo que se transfiere entre restaurantes son los **atributos** (cocina, ingredientes, proteína, picante, precio) y el **platillo canónico** ("Spaghetti alla carbonara" = `carbonara`). Los gustos se aprenden en ese espacio, no en el de los IDs.

**Criterio de éxito:** superar a la popularidad en el segmento **primera visita**, con una mejora mínima relevante definida **antes** del test y una potencia estadística verificada por simulación.

**Fuera del alcance (fase 2):** recomendar restaurantes; modelos profundos (NCF, RNN); el efecto de los acompañantes en "para ti".

## 2. Arquitectura del modelo

```
Comensal abre el menú
  → Candidatos = menú disponible a la fecha − agotados − alérgenos (suyos / de la mesa)
  → Perfil "a la fecha" (solo visitas anteriores)
  ├─ "Para ti":      ranking LightGBM LambdaRank · grupo = comensal-visita × tipo de plato
  └─ "Para la mesa": ranking sobre compartibles  · grupo = mesa-visita
```

### Variables

| Grupo | "Para ti" | "Para la mesa" |
|---|---|---|
| Perfil | Afinidad por cocina, proteína e ingredientes; picante; precio habitual; hábitos por tipo de plato | Agregado de los comensales registrados: **promedio** para el gusto, **miseria mínima** para el picante |
| Platillo | Atributos, precio relativo al menú, popularidad del restaurante y global del canónico | Lo mismo + `isShareable`, tasa de compartido |
| Coincidencia | ¿Pidió este canónico antes? ¿Cómo lo calificó? Jaccard de ingredientes, coseno perfil-platillo, diferencia de picante, ajuste de precio | **¿Alguien de la mesa suele pedirlo para compartir?** |
| Contexto | Hora (huso del restaurante), día, `pax` | `pax`, % de comensales con perfil conocido |

**Reglas de las variables:**

- **Todas se calculan "a la fecha":** popularidad, perfiles y estadísticas usan **solo** información anterior a la visita.
- **Se calculan con el mismo proceso que existirá en producción:** el canónico usado es el **predicho** por la homologación (KNN + confirmación simulada del restaurante), **nunca** el verdadero.
- **Alergias:** el filtro duro usa las del comensal ("para ti") o la **unión** de las de toda la mesa ("para la mesa").

### Modelos que compiten

1. **Jerarquía de respaldo:** popularidad del restaurante (si tiene al menos N pedidos anteriores; **N se elige en validación** y se guarda en el YAML) → popularidad global del canónico → contenido.
2. Similitud por contenido.
3. Regresión logística (el modelo base).
4. **LightGBM LambdaRank** (el principal), ajustado con Optuna.

## 3. Datos necesarios

### 3.1 Fuentes reales (públicas)

| Fuente | Para qué |
|---|---|
| **DENUE (INEGI)**, SCIAN 7225 | Restaurantes reales con ubicación y giro |
| **Censo / AGEB (INEGI)** | Población, para ubicar a los comensales |
| **ENIGH (INEGI)**, microdatos | Gasto en comida fuera del hogar por decil (**ponderado por el factor de expansión**) |
| **SEPOMEX** | Códigos postales y colonias |

### 3.2 Catálogos curados (sintéticos, declarados)

- **Biblioteca de ~220 platillos canónicos**, por cocina, con ingredientes, alérgenos, proteína, picante, tipo y si es compartible.
- **Alias** de los nombres (variaciones reales de la carta).
- Catálogos de ingredientes, alérgenos (NOM-051 / UE-14), etiquetas de dieta y cocinas.

### 3.3 Tablas simuladas

Son el contrato de datos = [`schema-recomendador.prisma`](schema-recomendador.prisma):

`restaurants`, `menu_items` (con historial de precio y disponibilidad), `diners`, `diner_profiles` (onboarding), `dining_sessions`, `guests` (con `diner_id`), `order_items` (**por comensal**, con `is_shared`), `dish_ratings`, `visit_ratings`, `recommendation_slates` (posición).

**`ground_truth/`** (solo la lee la evaluación): gustos latentes, utilidades, canónico verdadero, conjunto de consideración y la regla de agregación de cada mesa.

## 4. Simulación (Monte Carlo)

| Capa | Técnica |
|---|---|
| Mundo | Submuestreo estratificado del DENUE |
| Cocina del restaurante | Reglas sobre el nombre + **imputación con KNN** |
| Menús | Muestreo de la biblioteca + variaciones de nombre (ruido); precios **lognormales** por nivel |
| **Evolución del menú** | Altas, bajas, cambios de precio y agotados en el tiempo |
| Comensales: ubicación | **Muestreo con probabilidad proporcional a la población** de la AGEB + un punto dentro del polígono |
| Comensales: presupuesto | **Bootstrap suavizado** de la ENIGH, ponderado por el factor de expansión (la conversión a ticket por visita es un **supuesto declarado**) |
| Verdad oculta | Cocina (Dirichlet), proteínas, ingredientes que le gustan o evita, picante, λ de precio (lognormal), hábitos por tipo de plato (Beta), **rasgo "compartidor"** (Beta), deriva leve |
| Onboarding | Respuestas = versión **ruidosa e incompleta** de los gustos latentes |
| Visitas | Tiempo entre visitas Weibull + perfil horario empírico; restaurante por **logit mixto con Gumbel** (solo para simular) |
| Pedido individual | **Logit mixto** sobre el menú, por tipo de plato; alergias como filtro duro; **efecto de posición** configurable |
| Pedido para compartir | La probabilidad sube con `pax`; lo pide quien tiene el rasgo compartidor; la utilidad del grupo mezcla promedio y miseria mínima + Gumbel |
| Ticket | **Se calcula** con lo pedido |
| Calificaciones | Logit ordenado + faltantes no aleatorios (MNAR) |
| Ruido | GPS, onboarding incompleto (MAR), outliers, duplicados, pedidos del mesero sin comensal |
| Calibración | **Hipercubo latino** sobre la escala del Gumbel, λ y el peso de la afinidad |
| Validación | Hechos estilizados, Kolmogorov-Smirnov / Wasserstein, réplicas Monte Carlo |

**Escala por defecto** (configurable): 1 zona metropolitana, ~1,500 restaurantes, ~220 canónicos, 20,000 comensales, 12 meses.

## 5. Particiones y evaluación

```
Meses 1–9   TRAIN   validación cruzada temporal por origen móvil (Optuna)
Mes 10      VALID   selección de modelo, early stopping, umbral N
Meses 11–12 TEST    se usa UNA vez
```

- **Negativos:** los platillos del menú disponible **a la fecha** que no se pidieron.
- **Grupos sin ningún positivo:** fuera del entrenamiento. Las métricas son **condicionales**: "dado que pidió postre, ¿acertamos cuál?".
- **Métricas:**
  - NDCG@5, HitRate@3, Recall@5 y MAP@5 **por tipo de plato** y por lista.
  - Cobertura y novedad.
  - **Techo del oráculo** (el ranking según la utilidad verdadera).
- **Segmentos:**
  - **Primera visita** (el principal).
  - Visita repetida.
  - Restaurante nuevo en la plataforma.
  - **Platillo nuevo.**
  - Comensal sin historial (con y sin onboarding).
- **Pruebas estadísticas:** **permutación** y **bootstrap pareado por comensal**, con intervalos de confianza bootstrap.
- **Análisis de potencia con Monte Carlo** antes del test: réplicas con una mejora conocida → porcentaje en que la prueba la detecta.

## 6. Cobertura de la rúbrica

| Requisito | Dónde |
|---|---|
| CRISP-DM | Los notebooks 00–15 siguen sus 6 fases |
| EDA uni-, bi- y multivariado | 06–08: pruebas no paramétricas, MCA, clustering de comensales |
| Calidad de datos | 05: completitud, unicidad, validez, consistencia (ticket = suma de platillos), outliers |
| Faltantes | 10: mediana, KNN e indicador de faltante. **MNAR solo se puede verificar porque se simuló** |
| One-hot / encoding | 10: one-hot para baja cardinalidad, *target encoding* por fold para la alta |
| KNN | 01 (cocina del restaurante) y 09 (**homologación de platillos**, con exactitud medida contra la verdad) |
| Separación de datos | 10: partición temporal, grupos y segmentos |
| Modelo base y competencia | 11–12 |
| Hiperparámetros (bayesiano) | 12: **Optuna TPE**, historial e importancia de cada hiperparámetro |
| Sobreajuste y validación | 13: curvas de aprendizaje, brecha entre entrenamiento y validación, pruebas pareadas, potencia |
| Variables importantes / SHAP | 14: TreeExplainer. Contra la verdad se compara **signo, orden y forma** (no magnitudes). Se declara el reparto entre variables correlacionadas |
| Lectura de negocio | 14–15 |

## 7. Estructura del proyecto

```
Recomendador/
├── config/                paths.yaml · simulation.yaml · features.yaml · models.yaml
├── data/                  (fuera de git)
│   ├── raw/               INEGI, DENUE, ENIGH, SEPOMEX tal cual
│   ├── external/          biblioteca canónica, alias, catálogos
│   ├── interim/           mundo limpio y homologado
│   ├── synthetic/<run_id>/  tablas del contrato + config usada
│   │   └── ground_truth/
│   ├── processed/         tablas de features ("para ti" y "para la mesa")
│   └── Data_model/        X_, y_, group_ de train · valid · test + segmentos
├── docs/                  propuesta.md · cambios-base-de-datos.md · adr/ ·
│                          diccionario_datos.md · supuestos.md · model_card.md
├── models/                artefactos versionados
├── mlruns/                MLflow
├── notebooks/
│   00_entendimiento_negocio          08_eda_multivariado
│   01_fuentes_y_mundo                09_homologacion_y_perfil_comensal
│   02_biblioteca_platillos_y_menus   10_preparacion_y_particion
│   03_simulacion_montecarlo          11_modelo_base
│   04_validacion_simulacion          12_lightgbm_ranker_optuna
│   05_calidad_datos                  13_validacion_modelo
│   06_eda_univariado                 14_interpretabilidad_shap
│   07_eda_bivariado                  15_evaluacion_final
├── reports/figures/
├── src/puzzle_recsys/
│   ├── config.py          carga + validación (pydantic)
│   ├── repository/        puerto Repository: base.py · parquet_repo.py · postgres_repo.py
│   ├── sources/           inegi.py · denue.py · enigh.py · sepomex.py
│   ├── simulation/        distributions.py · world.py · menus.py · population.py ·
│   │                      choice.py · group_choice.py · engine.py · feedback.py ·
│   │                      noise.py · calibration.py
│   ├── quality/           checks.py · report.py
│   ├── eda/               univariate.py · bivariate.py · multivariate.py · plots.py
│   ├── catalog/           dish_homologation.py · cuisine_imputation.py
│   ├── profiles/          diner_profile.py · table_profile.py   ("a la fecha")
│   ├── features/          build.py · encoding.py · missing.py
│   ├── splitting/         temporal.py · segments.py
│   ├── ranking/           baselines.py · logistic.py · lgbm_ranker.py
│   ├── tuning/            optuna_search.py
│   ├── evaluation/        metrics.py · oracle.py · resampling.py ·
│   │                      statistical_tests.py · power.py
│   ├── explainability/    shap_analysis.py · ground_truth_check.py
│   └── pipelines/         simulate.py · build_dataset.py · train.py · score.py (CLI)
├── tests/                 mismo árbol que src
└── pyproject.toml · uv.lock · .env.example · .gitignore · .pre-commit-config.yaml · README.md
```

## 8. Buenas prácticas

- **Notebooks = el algoritmo, paso a paso:** markdown con el qué y el por qué, y celdas cortas que llaman a `src`. La lógica vive en `src`.
- **Pipelines por línea de comandos** que corren lo mismo que los notebooks. Es lo que usa producción.
- **Entorno y estilo:** `uv` + `pyproject.toml` con archivo de bloqueo; `ruff`; *type hints*; docstrings estilo NumPy con los supuestos estadísticos.
- **Sin números mágicos:** todo parámetro vive en YAML.
- **Tests:** `pytest` + `hypothesis` (tests de propiedades):
  - La Dirichlet suma 1.
  - El ticket es igual a la suma de los platillos.
  - Ningún alérgico recibe su alérgeno, ni para él ni para su mesa.
  - Ningún perfil ni popularidad usa datos posteriores a la visita.
  - Las variables usan el canónico **predicho**.
- **Reproducibilidad:** un `numpy.random.Generator` que se pasa como argumento, `run_id` + copia de la configuración, MLflow y `pre-commit`.
- **Documentación:** ADRs, diccionario de datos, supuestos y *model card*.

## 9. Llegada a Puzzle

1. Se aplican los cambios de [`cambios-base-de-datos.md`](cambios-base-de-datos.md) (P0 primero).
2. `postgres_repo.py` reemplaza a `parquet_repo.py`. El resto del código no cambia.
3. El perfil se calcula de noche; el ranking, **al abrir el menú** (servicio chico en Python, unos 50 platillos por consulta).
4. Cada lista mostrada se guarda en `RecommendationSlate` con su posición. Un **grupo de control** (5–10 %, por popularidad) mide la mejora real con una prueba A/B.
5. **Fase 2:** reentrenar con datos reales, corregir el sesgo de posición (*propensity scoring*), recomendar restaurantes y evaluar NCF y RNN.

## 10. Alcance según la fecha de entrega

| Núcleo (entra siempre) | Si alcanza el tiempo | Fase 2 |
|---|---|---|
| Simulación completa, EDA, calidad, particiones, "para ti" con los 4 modelos, Optuna, SHAP, evaluación por segmentos | Ranker completo de "para la mesa" (si no, va con la línea base: popularidad de compartibles + hábito de la mesa), análisis de potencia, calibración por hipercubo latino | Restaurantes, NCF y RNN, corrección del sesgo de posición, acompañantes |

**Zona simulada:** CDMX (16 alcaldías). **Pendiente:** la fecha de entrega.
