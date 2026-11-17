# Cambios a la base de datos para el recomendador

**Para:** Diego
**De:** Natalia
**Fecha:** 2026-11-16
**Estado:** propuesta. No se aplicó nada en `bouquet`.
**Esquema completo:** [`schema-recomendador.prisma`](schema-recomendador.prisma). Ya está validado con `prisma validate` junto con el esquema actual; el final del archivo trae, como comentario, los campos que se agregan a los modelos existentes.

## Para qué

Puzzle va a recomendar **platillos** a comensales registrados en dos listas: **"para ti"** y **"para la mesa"**. Funciona sobre todo cuando el comensal entra a un restaurante **por primera vez**. Para eso la base tiene que guardar tres cosas que hoy no guarda:

- **Quién es el comensal**, de forma persistente entre restaurantes.
- **Qué es cada platillo**: su platillo canónico y sus atributos.
- **Qué había en el menú** en cada visita: precio y disponibilidad *a la fecha*.

## 1. Lo que encontré en el esquema actual

| # | Hoy | Problema para el recomendador | Cambio |
|---|---|---|---|
| 1 | `Guest` es anónimo y vive dentro de una sola sesión | No hay historial por persona entre restaurantes | `Diner` + `DinerIdentity` (correo o celular, código de un solo uso) + `Guest.dinerId` |
| 2 | Las estrellas de `CheckoutSuccessScreen.tsx` (`handleSubmitRating`) **no se guardan** | Se pierde la señal explícita | `VisitRating` y `DishRating` |
| 3 | `OrderItem.guestId` es opcional | Los pedidos que carga el mesero sin asignar comensal no se pueden usar para aprender gustos | Que el mesero elija el comensal o marque "para la mesa" |
| 4 | No existe la noción de "para compartir" | El guacamole de la mesa ensucia el perfil de quien lo pidió | `OrderItem.isShared` |
| 5 | `RestaurantMenuItem.templateItemId` es opcional | Si los atributos vivieran solo en `TemplateItem`, los platillos sin plantilla quedarían sin etiquetar | Atributos en el **platillo servido** (`RestaurantMenuItem`); el canónico da los valores por defecto |
| 6 | Sin tipo de cocina, ingredientes ni alérgenos | No hay forma de transferir gustos ni de filtrar alergias | `CanonicalDish` + `MenuItemIngredient` / `MenuItemAllergen` / `MenuItemDietaryTag` |
| 7 | `RestaurantMenuItem.priceCents` se **sobrescribe** | No se conoce el precio *a la fecha* de los platillos **no** pedidos (`OrderItem` solo guarda el de los pedidos) | `MenuItemPriceHistory` |
| 8 | `isSoldOut` guarda solo el estado **actual** | No se sabe qué había disponible en cada visita | `MenuItemAvailabilityEvent` |
| 9 | `RestaurantCategory.name` es texto libre ("Para empezar", "Entradas") | No se sabe si un platillo es entrada, fuerte, bebida o postre | `RestaurantMenuItem.course` (enum) |
| 10 | Los modificadores son texto, sin vínculo con ingredientes | "Sin cebolla" es una preferencia explícita que se pierde | `ModifierOption.ingredientId` + `effect` |
| 11 | `Restaurant.address` es texto, sin zona horaria | Las variables de hora quedan mal en estados con otro huso | `Address` estructurada + `Restaurant.timezone` |
| 12 | No se registra qué se mostró ni en qué posición | Imposible medir el efecto de la posición o hacer una prueba A/B | `RecommendationSlate` / `RecommendationSlateItem` + `ExperimentAssignment` |
| 13 | Los estados son `String` (`OrderItem.status = "CANCELADA"`) | Hay riesgo de valores inconsistentes; el modelo tiene que excluir lo cancelado | Recomendado: pasarlos a `enum` (no bloquea) |
| 14 | `DiningSession.pax` tiene valor por defecto 1 | Puede no coincidir con el número real de `Guest` | Validarlo al cerrar la sesión (no bloquea) |
| 15 | `AnalyticItemVelocity` se recalcula | No garantiza valores *a la fecha* | El modelo calcula la popularidad desde `OrderItem` (sin cambio de esquema) |

## 2. Prioridades

**P0: sin esto no hay recomendador**

- `Diner`, `DinerIdentity`, `DinerVerificationCode`, `DinerConsent`, `DinerProfile` + `Guest.dinerId`.
- `OrderItem.isShared`.
- `VisitRating`, `DishRating` (y guardar las estrellas que ya existen en la UI).
- `CanonicalDish`, `CanonicalDishAlias` + los campos de canónico en `TemplateItem` y `RestaurantMenuItem`.
- `MenuItemIngredient`, `MenuItemAllergen`, `MenuItemDietaryTag` + catálogos `Ingredient`, `Allergen`, `DietaryTag`, `Cuisine`.
- `RestaurantMenuItem.course`, `protein`, `spiceLevel`, `isShareable`.
- `MenuItemPriceHistory`, `MenuItemAvailabilityEvent`.
- `Restaurant.timezone`.
- Preferencias del onboarding: `DinerCuisinePreference`, `DinerDietaryTag`, `DinerAllergen`.

**P1: para medir y mejorar en producción**

- `RecommendationSlate`, `RecommendationSlateItem`, `ExperimentAssignment`.
- `InteractionEvent`.
- `ModifierOption.ingredientId` + `effect`.
- `MenuItemStats`.

**P2: recomendación de restaurantes (fase 2)**

- Geografía: `Country`, `CountryConfig`, `AdminArea1/2`, `PostalCode`, `Neighborhood`, `Address`, `PriceTier`.
- `RestaurantCuisine`, `RestaurantAttribute`, `RestaurantHours`, `RestaurantPhoto`, `RestaurantStats`, `DinerAffinity`.

## 3. Reglas para que los datos sirvan

1. **Historial en la misma transacción.** Cada cambio de `priceCents` o `isSoldOut` escribe su fila en `MenuItemPriceHistory` o `MenuItemAvailabilityEvent` **dentro de la misma transacción**. Si se escribe aparte, el historial se desincroniza.
2. **Nunca borrar platillos**, solo archivarlos (`archivedAt`). Los pedidos y el historial apuntan a ellos.
3. **Alérgenos obligatorios** al crear un platillo. Se proponen los del canónico y el restaurante los confirma.
4. **El canónico se sugiere, no se impone.** El KNN propone y el restaurante confirma (`canonicalSource = CONFIRMED`) o corrige (`MANUAL`). Cada confirmación agrega un `CanonicalDishAlias`.
5. **Fechas en UTC**, y para las variables de hora se interpretan con `Restaurant.timezone`.
6. **Identidades normalizadas:** teléfono en E.164 (`+52` + 10 dígitos) y correo en minúsculas.
7. **La asignación al experimento es estable:** un comensal no cambia de grupo entre visitas.
8. **Las alergias son un filtro duro** en el servidor, nunca solo en la UI.

## 4. Cambios de producto que acompañan

- **Comensal:**
  - Registro con correo o celular.
  - Onboarding de 3–4 preguntas saltables: picante, dieta, alergias, cocinas favoritas.
  - Interruptor **"¿Es para compartir?"** al pedir.
  - Guardar las calificaciones.
- **Mesero:** al cargar un pedido, elegir el comensal o "para la mesa".
- **Restaurante (alta del menú):**
  - Confirmar el platillo canónico sugerido.
  - Confirmar alérgenos e ingredientes.
  - Elegir el tipo de plato.
- **Menú:** mostrar las listas "Para ti" y "Para la mesa", registradas en `RecommendationSlate`.

## 5. Migración de los datos existentes

| Tabla nueva | Cómo llenarla con lo que ya hay |
|---|---|
| `MenuItemPriceHistory` | Una fila por platillo con el `priceCents` actual, desde `createdAt` |
| `MenuItemAvailabilityEvent` | `CREATED` en `createdAt`; `ARCHIVED` en `archivedAt` si existe |
| Canónico de los platillos actuales | Correr la homologación (KNN) → `SUGGESTED`; el restaurante confirma después |
| `Guest.dinerId` | Queda vacío: las visitas anónimas del pasado no se pueden reclamar |
| `RestaurantMenuItem.course` | Sugerido a partir del nombre de la categoría; el restaurante lo confirma |

## 6. Preguntas para vos

1. ¿Preferís el historial de precio y disponibilidad en la aplicación (en la misma transacción) o con *triggers* de Postgres?
2. ¿Pasamos los estados `String` a `enum` en esta migración o en una aparte?
3. ¿El servicio del recomendador (Python) lee de la base principal o de una réplica de solo lectura?
