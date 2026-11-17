# Recomendador de platillos de Puzzle

Proyecto doble:

- **Entrega del curso:** CRISP-DM con datos sintéticos.
- **Módulo de recomendación de la app Puzzle.**

Cuando un comensal registrado entra a un restaurante, aunque sea la primera vez, recibe dos listas ordenadas según sus gustos: **"para ti"** y **"para la mesa"**.

## Documentos

| Documento | Contenido |
|---|---|
| [`docs/propuesta.md`](docs/propuesta.md) | Propuesta técnica completa |
| [`docs/cambios-base-de-datos.md`](docs/cambios-base-de-datos.md) | Qué le falta a la base de Puzzle, con prioridades (para Diego) |
| [`docs/schema-recomendador.prisma`](docs/schema-recomendador.prisma) | Esquema propuesto (v2), validado con Prisma |
| [`docs/adr/`](docs/adr/README.md) | Decisiones de arquitectura |
| [`docs/supuestos.md`](docs/supuestos.md) | Todo lo que no sale de datos reales |
| [`docs/diccionario_datos.md`](docs/diccionario_datos.md) | Contrato de las tablas |
| [`docs/model_card.md`](docs/model_card.md) | Uso, límites y sesgos del modelo |

## Estructura

```
config/       parámetros (YAML): rutas, simulación, features, modelos
data/         raw · external · interim · synthetic/<run_id> · processed · Data_model   (fuera de git)
docs/         propuesta, esquema, ADRs, supuestos, diccionario, model card
models/       artefactos entrenados
notebooks/    00–15: el algoritmo paso a paso, por fases de CRISP-DM
reports/      figuras
src/puzzle_recsys/   toda la lógica (los notebooks solo la llaman)
tests/        pytest + hypothesis, mismo árbol que src
```

## Instalación

Requiere **Python 3.13** (está fijado en `.python-version`). Con [`uv`](https://docs.astral.sh/uv/), que es lo recomendado:

```powershell
winget install astral-sh.uv          # una sola vez
uv sync --extra dev --extra notebooks # crea .venv con las versiones EXACTAS de uv.lock
uv run pre-commit install
```

Sin `uv`, se puede usar `requirements.txt`, que se exporta desde `uv.lock`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Si agregás una dependencia: `uv add <paquete>` y después
`uv export --format requirements-txt --no-hashes --extra dev --extra notebooks -o requirements.txt`.

## Uso

```powershell
uv run pytest                         # tests
uv run ruff check . ; uv run ruff format .
uv run jupyter lab                    # notebooks en orden: 00 → 15
```

> Si tenés Anaconda activo, `uv` avisa que ignora `VIRTUAL_ENV`. Es normal: usa el `.venv` del proyecto.

## Reglas del proyecto

1. Los notebooks **narran**: markdown con el qué y el por qué, y celdas cortas que llaman a `src`.
2. **Sin números mágicos:** todo parámetro vive en `config/`.
3. **Reproducible:** cada función aleatoria recibe un `numpy.random.Generator`, y cada corrida guarda su configuración.
4. **Sin fugas:** las variables se calculan "a la fecha", usan el canónico **predicho**, y `ground_truth/` solo lo lee la evaluación.
5. **Tests** para toda función de `src`.

**Zona simulada:** CDMX (16 alcaldías). **Pendiente:** fecha de entrega.
