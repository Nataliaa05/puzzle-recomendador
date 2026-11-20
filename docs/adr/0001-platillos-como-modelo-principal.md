# 0001 · Platillos como modelo principal, con dos listas

**Estado:** Aceptada (2026-11-16)

## Contexto

El caso de uso de Puzzle es que un comensal registrado entra a un restaurante, muchas veces por primera vez, y quiere saber qué pedir según sus gustos. En la mesa conviven pedidos individuales y platillos para compartir (por ejemplo, el guacamole de la mesa). Si se mezclan, el platillo compartido ensucia el perfil de quien lo pidió.

## Decisión

- El modelo principal recomienda **platillos dentro de un restaurante**.
- Se producen dos listas:
  - **"Para ti":** ranking individual, con grupo = comensal-visita × tipo de plato.
  - **"Para la mesa":** ranking de compartibles, con grupo = mesa-visita. Un platillo para compartir se reparte **solo entre quienes pueden comerlo** (sin sus alérgenos y compatible con su dieta; al menos 2 personas). La utilidad del grupo se calcula con esas personas (promedio + miseria mínima) y se resta un costo social por dejar fuera a alguien. En el producto, la recomendación lo indica: *"Contiene camarón: no apto para Ana"*.
- Recomendar restaurantes pasa a la fase 2.

## Consecuencias

- Hace falta `OrderItem.isShared` y que el mesero asigne el comensal o marque "para la mesa".
- Las métricas son **condicionales** por tipo de plato: "dado que pidió postre, ¿acertamos cuál?".
- Si falta tiempo, "para la mesa" se entrega con la línea base (popularidad de compartibles + hábito de compartir de la mesa).

## Revisión (2026-11-18)

La primera versión aplicaba la **unión de las alergias de toda la mesa** como filtro duro para lo que se comparte. Era demasiado estricta: si una persona es alérgica al camarón, los demás igual pueden compartir un aguachile y ella no come de ese plato. Se cambió a "reparto entre quienes pueden comerlo". La regla individual ("para ti") no cambia: nunca se recomienda a alguien un platillo con sus alérgenos. Se reconoce el riesgo de **contacto cruzado** en platos al centro de la mesa; por eso el producto debe advertir explícitamente a quién excluye cada recomendación.
