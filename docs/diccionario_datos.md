# Diccionario de datos

Contrato de las tablas que lee el modelo (snake_case de [`schema-recomendador.prisma`](schema-recomendador.prisma)). Se completa en el notebook 03, cuando la simulación genera cada tabla.

| Tabla | Grano | Columnas clave | Origen |
|---|---|---|---|
| `restaurants` | 1 restaurante | `restaurant_id`, `cuisine_pred`, `lat`, `lng`, `adminarea2`, `timezone` | DENUE + homologación |
| `canonical_dishes` | 1 platillo canónico | `canonical_id`, `cuisine`, `course`, `protein`, `spice_level`, `is_shareable` | Biblioteca |
| `canonical_dish_aliases` | 1 alias | `canonical_id`, `alias` | Biblioteca |
| `menu_items` | 1 platillo servido | `menu_item_id`, `restaurant_id`, `name`, `canonical_pred`, `canonical_source`, `course` | Simulación |
| `menu_item_attributes` | platillo × atributo | `menu_item_id`, `kind` (ingredient/allergen/diet), `value` | Simulación |
| `menu_item_price_history` | platillo × vigencia | `menu_item_id`, `price_cents`, `valid_from`, `valid_to` | Simulación |
| `menu_item_availability_events` | 1 evento | `menu_item_id`, `is_available`, `reason`, `occurred_at` | Simulación |
| `diners` | 1 comensal registrado | `diner_id`, `created_at` | Simulación |
| `diner_profiles` | 1 comensal | respuestas del onboarding (con ruido y faltantes) | Simulación |
| `dining_sessions` | 1 visita de mesa | `session_id`, `restaurant_id`, `pax`, `opened_at` | Simulación |
| `guests` | comensal × visita | `guest_id`, `session_id`, `diner_id` | Simulación |
| `order_items` | 1 platillo pedido | `guest_id`, `menu_item_id`, `quantity`, `unit_price_cents`, `is_shared`, `created_at` | Simulación |
| `dish_ratings` / `visit_ratings` | 1 calificación | `stars`, `created_at` | Simulación |
| `recommendation_slates` | lista mostrada | `list_type`, `course`, `position` | Simulación |

`ground_truth/` (fuera del contrato): `diner_tastes`, `utilities`, `canonical_true`, `consideration_sets`, `table_aggregation`.
