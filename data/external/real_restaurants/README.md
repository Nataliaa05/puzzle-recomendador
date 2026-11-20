# Base de datos real de restaurantes y menús

Información **real** de restaurantes de la CDMX, recolectada a mano o con apoyo automático y **verificada**. Su función es anclar la simulación a la realidad:

- **Calibrar precios** por cocina y tipo de plato (bootstrap con una semilla real, en lugar de suponer distribuciones).
- **Probar la homologación** de platillos (notebook 09) con nombres de carta reales.
- **Ampliar la biblioteca canónica** y sus alias.

## Archivos

| Archivo | Grano | Contenido |
|---|---|---|
| `restaurants.csv` | 1 restaurante | Ficha: id del DENUE, dirección, coordenadas, web, URL del menú, formato, cocina observada, estado del menú, fuentes |
| `menu_items.csv` | 1 precio de un platillo | Sección, nombre, descripción, precio, unidad (`PLATILLO`, `COPA`, `BOTELLA`), tipo de plato y la imagen de origen |

Las imágenes originales se guardan en `data/raw/menus/<restaurant_key>/` (fuera de git), para poder auditar cada transcripción.

## Estado de cada restaurante (`menu_status`)

| Valor | Significado |
|---|---|
| `TRANSCRIBED` | Menú completo transcrito y verificado contra la fuente |
| `NO_MENU_FOUND` | El restaurante existe, pero no hay un menú público con precios en una fuente permitida |

## Reglas de recolección

1. **Solo fuentes permitidas:** el sitio propio del restaurante, respetando `robots.txt`, a un ritmo lento y con un agente identificado. **No** se extraen datos de Google Maps, Instagram ni Facebook, porque sus términos lo prohíben.
2. **Solo hechos:** nombre del platillo, descripción breve y precio. No se republican imágenes.
3. **Verificación de identidad:** la coincidencia con el DENUE se confirma por **dirección**, no solo por nombre. Por ejemplo, `urso.mx` resultó ser una empresa de ingeniería eléctrica, no el restaurante URSO.
4. **Trazabilidad:** cada fila guarda su fuente, y cada restaurante su fecha de captura. Los precios cambian, así que esto es una **foto en el tiempo**.

## Hallazgos del primer caso (Salvaje)

- **El menú venía en imágenes**, no en texto. El HTML del sitio no tenía ningún precio, así que hay que leer las imágenes (OCR o transcripción).
- **La URL del DENUE era inválida** (`www.salvaje_mx`); el sitio real es `salvaje.art`.
- **La cocina observada (japonesa/fusión) contradice la imputada** por el pipeline (`mexicana` por default, con confianza 0.26). Es evidencia directa de la limitación declarada en el supuesto S9.
