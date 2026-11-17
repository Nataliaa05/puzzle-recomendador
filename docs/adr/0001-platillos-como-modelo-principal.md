# 0001 · Platillos como modelo principal, con dos listas

**Estado:** Aceptada (2026-11-16)

## Contexto

El caso de uso de Puzzle es que un comensal registrado entra a un restaurante, muchas veces por primera vez, y quiere saber qué pedir según sus gustos. En la mesa conviven pedidos individuales y platillos para compartir (por ejemplo, el guacamole de la mesa). Si se mezclan, el platillo compartido ensucia el perfil de quien lo pidió.

## Decisión

- El modelo principal recomienda **platillos dentro de un restaurante**.
- Se producen dos listas:
  - **"Para ti":** ranking individual, con grupo = comensal-visita × tipo de plato.
  - **"Para la mesa":** ranking de compartibles, con grupo = mesa-visita. El perfil de la mesa agrega a sus comensales: promedio para el gusto, miseria mínima para el picante y unión de alergias como filtro duro.
- Recomendar restaurantes pasa a la fase 2.

## Consecuencias

- Hace falta `OrderItem.isShared` y que el mesero asigne el comensal o marque "para la mesa".
- Las métricas son **condicionales** por tipo de plato: "dado que pidió postre, ¿acertamos cuál?".
- Si falta tiempo, "para la mesa" se entrega con la línea base (popularidad de compartibles + hábito de compartir de la mesa).
