# 0004 · LightGBM LambdaRank; sin redes neuronales ni AutoML en la fase 1

**Estado:** Aceptada (2026-11-16)

## Contexto

El ranking de platillos se formula como un problema tabular (comensal × platillo × contexto). En datos tabulares, los árboles con *boosting* suelen ser muy competitivos. Además, la literatura de recomendación mostró que varios modelos neuronales no superaban a líneas base bien ajustadas: Dacrema et al., RecSys 2019, y Rendle et al., RecSys 2020, sobre NCF frente a la factorización de matrices.

## Decisión

- Los modelos que compiten son:
  - La jerarquía de popularidad.
  - Similitud por contenido.
  - Regresión logística.
  - **LightGBM LambdaRank**, ajustado con Optuna (TPE).
- Sin PyCaret ni H2O: agregan dependencias (versiones viejas, Java) y una comparación que no aporta a la rúbrica.
- NCF, Wide & Deep y RNN pasan a la fase 2, con datos reales.

## Consecuencias

- El entorno es más liviano: no hace falta PyTorch ni Java.
- SHAP TreeExplainer es exacto y rápido sobre LightGBM.
