"""Piloto de extracción de menús desde los sitios web propios de los restaurantes.

Objetivo: medir **qué fracción** de los restaurantes del DENUE con sitio web
publica un menú con precios que se pueda leer, y en qué formato (texto, PDF o
imágenes). Con ese número se decide si los menús reales se integran como
semilla de la simulación.

Reglas (legales y de cortesía), aplicadas en el código:

1. **Solo el sitio propio del restaurante.** Nunca Google, Instagram,
   Facebook ni plataformas de entrega (:data:`BLOCKED_DOMAINS`): sus términos
   prohíben la extracción automática.
2. **Se respeta ``robots.txt``**: si el sitio pide no ser leído, no se pide
   ninguna página.
3. **Pausa entre peticiones** al mismo dominio y agente identificado.
4. **Solo hechos** (nombre del platillo y precio). No se guardan imágenes.

Usado en: notebooks 02 (piloto de menús reales).
"""

from __future__ import annotations

import html as html_lib
import http.client
import io
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from typing import Any, Protocol

USER_AGENT = "puzzle-recsys/0.1 (proyecto academico; contacto en el README)"

#: Plataformas cuyos términos prohíben la extracción automática: nunca se tocan.
BLOCKED_DOMAINS: tuple[str, ...] = (
    "google.com",
    "goo.gl",
    "instagram.com",
    "facebook.com",
    "fb.com",
    "tiktok.com",
    "rappi.com.mx",
    "ubereats.com",
    "didi-food.com",
)

_DOMAIN = re.compile(r"^[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}$")
_MENU_HINT = re.compile(r"men[uú]|carta|desayun|comida|bebida|drinks|food|platillo|\.pdf", re.I)
_PRICE_AT_END = re.compile(
    r"^(?P<name>.*?[A-Za-zÁÉÍÓÚÑáéíóúñ].*?)\s*\$?\s*(?P<price>\d{2,4}(?:[.,]\d{2})?)\s*$"
)
_NOT_MENU = re.compile(
    r"aviso|privacidad|legal|t[eé]rminos|condiciones|factura|empleo|vacante", re.I
)
_HREF = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.I | re.S)
_IMG = re.compile(r"<img\b[^>]*src=[\"']([^\"']+)[\"']", re.I)
_MIN_ITEMS = 5
_MIN_IMAGES = 4


def _is_blocked(host: str) -> bool:
    host = host.lower()
    return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)


def normalize_website(raw: str | None) -> str | None:
    """Convierte el campo ``www`` del DENUE en una URL utilizable, o ``None``.

    Corrige espacios sueltos (``"maison-kayser.com .mx"``), agrega ``https://``
    y descarta dominios inválidos (``"www.salvaje_mx"``) y redes sociales.
    """
    if raw is None or not str(raw).strip():
        return None
    text = str(raw).strip().lower().replace(" ", "")
    if not text.startswith(("http://", "https://")):
        text = "https://" + text
    parsed = urllib.parse.urlparse(text)
    host = parsed.netloc
    if not _DOMAIN.match(host) or _is_blocked(host):
        return None
    return text.rstrip("/") if parsed.path in ("", "/") else text


def html_to_text(page: str) -> str:
    """Texto visible de una página: sin scripts ni estilos, una línea por bloque."""
    body = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", page)
    body = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h\d)>", "\n", body)
    text = html_lib.unescape(re.sub(r"(?s)<[^>]+>", " ", body))
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def find_menu_links(page: str, base_url: str, *, limit: int = 6) -> list[str]:
    """Enlaces del mismo sitio que parecen menú (por texto o URL) o PDF."""
    base_host = urllib.parse.urlparse(base_url).netloc.removeprefix("www.")
    found: list[str] = []
    for href, label in _HREF.findall(page):
        # Codifica espacios y caracteres raros ("Menu 2025.pdf" → "Menu%202025.pdf").
        url = urllib.parse.quote(
            urllib.parse.urljoin(base_url, href.strip()), safe=":/?&=%#~+,;@!$'()*"
        )
        text = re.sub(r"<[^>]+>", " ", label)
        if _NOT_MENU.search(href) or _NOT_MENU.search(text):  # avisos legales, términos, etc.
            continue
        host = urllib.parse.urlparse(url).netloc.removeprefix("www.")
        if host != base_host or _is_blocked(host) or url in found:
            continue
        if _MENU_HINT.search(href) or _MENU_HINT.search(text):
            found.append(url)
    return found[:limit]


def extract_price_lines(text: str) -> list[tuple[str, float]]:
    """Pares (nombre, precio) de líneas que terminan en un precio plausible."""
    items: list[tuple[str, float]] = []
    for raw in text.splitlines():
        line = raw.strip()
        match = _PRICE_AT_END.match(line)
        if not match:
            continue
        name = match["name"].strip(" .:-$")
        price = float(match["price"].replace(",", "."))
        if not 3 <= len(name) <= 90 or not 10 <= price <= 5000:
            continue
        if re.search(r"\b(tel|horario|am|pm|cp|c\.p\.|col\.)\b", name, re.I):
            continue
        items.append((name, price))
    return items


def pdf_to_text(content: bytes) -> str:
    """Texto de un PDF (requiere ``pypdf``)."""
    from pypdf import PdfReader  # import perezoso: solo si aparece un PDF

    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


class Fetcher(Protocol):
    """Contrato mínimo de acceso a la web (permite probar sin internet)."""

    def allowed(self, url: str) -> bool:
        """``robots.txt`` permite leer ``url``."""
        ...

    def get(self, url: str) -> tuple[str, Any] | None:
        """``(content_type, cuerpo)`` o ``None`` si falla."""
        ...


class PoliteFetcher:
    """Cliente HTTP que respeta ``robots.txt`` y pausa entre peticiones por dominio."""

    def __init__(self, *, delay_s: float = 2.0, timeout_s: float = 20.0) -> None:
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last: dict[str, float] = {}

    def allowed(self, url: str) -> bool:
        """Consulta (y cachea) el ``robots.txt`` del dominio."""
        parsed = urllib.parse.urlparse(url)
        if _is_blocked(parsed.netloc):
            return False
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self._robots:
            rp = urllib.robotparser.RobotFileParser(root + "/robots.txt")
            try:
                rp.read()
            except OSError:
                rp = None  # sin robots.txt accesible: se permite (convención)
            self._robots[root] = rp
        rp = self._robots[root]
        return True if rp is None else rp.can_fetch(USER_AGENT, url)

    def get(self, url: str) -> tuple[str, Any] | None:
        """Descarga ``url`` respetando la pausa por dominio."""
        host = urllib.parse.urlparse(url).netloc
        wait = self.delay_s - (time.monotonic() - self._last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                ctype = response.headers.get_content_type()
                body = response.read(15_000_000)
        except (OSError, ValueError, http.client.HTTPException):
            return None
        finally:
            self._last[host] = time.monotonic()
        if ctype == "application/pdf" or body[:4] == b"%PDF":
            return "application/pdf", body
        return ctype, body.decode("utf-8", errors="replace")


def _parse(ctype: str, body: Any) -> dict[str, Any]:
    """Platillos con precio e imágenes de una página (HTML) o documento (PDF)."""
    if ctype == "application/pdf":
        try:
            items = extract_price_lines(pdf_to_text(body))
        except Exception:
            items = []
        return {"format": "PDF", "items": items, "n_images": 0}
    return {
        "format": "HTML",
        "items": extract_price_lines(html_to_text(body)),
        "n_images": len(_IMG.findall(body)),
    }


def _safe_allowed(fetcher: Fetcher, url: str) -> bool:
    """``robots.txt`` permite ``url``; ante cualquier error, por prudencia, NO."""
    try:
        return fetcher.allowed(url)
    except Exception:
        return False


def _safe_get(fetcher: Fetcher, url: str) -> tuple[str, Any] | None:
    """Descarga ``url``; ante cualquier error, ``None`` (un sitio raro no tumba el piloto)."""
    try:
        return fetcher.get(url)
    except Exception:
        return None


def _read_menu(fetcher: Fetcher, url: str) -> dict[str, Any] | None:
    """Descarga y analiza ``url`` si ``robots.txt`` lo permite."""
    if not _safe_allowed(fetcher, url):
        return None
    got = _safe_get(fetcher, url)
    if got is None:
        return None
    try:
        return _parse(*got)
    except Exception:
        return None


def _is_better(new: dict[str, Any], current: dict[str, Any]) -> bool:
    """Más platillos con precio gana; a igualdad sin precios, más imágenes (menú en fotos)."""
    if len(new["items"]) != len(current["items"]):
        return len(new["items"]) > len(current["items"])
    return not new["items"] and new["n_images"] > current["n_images"]


def pilot_restaurant(fetcher: Fetcher, restaurant_id: str, website: str | None) -> dict[str, Any]:
    """Intenta encontrar y leer el menú de un restaurante. Nunca lanza excepciones.

    Pasos: normalizar la URL → respetar ``robots.txt`` → leer la portada →
    seguir los enlaces que parecen menú (o PDF) → quedarse con la página que
    tenga más platillos con precio.

    Returns
    -------
    dict
        ``restaurant_id``, ``url``, ``status`` (``INVALID_URL``,
        ``ROBOTS_DISALLOW``, ``UNREACHABLE``, ``NO_MENU_LINK``, ``MENU_TEXT``,
        ``MENU_PDF``, ``MENU_IMAGES``, ``MENU_NO_PRICES``), ``menu_url``,
        ``n_items`` e ``items`` (lista de (nombre, precio)).
    """
    url = normalize_website(website)
    out: dict[str, Any] = {
        "restaurant_id": restaurant_id,
        "url": url,
        "status": "INVALID_URL",
        "menu_url": None,
        "n_items": 0,
        "items": [],
    }
    if url is None:
        return out
    if not _safe_allowed(fetcher, url):
        out["status"] = "ROBOTS_DISALLOW"
        return out
    home = _safe_get(fetcher, url)
    if home is None:
        out["status"] = "UNREACHABLE"
        return out

    ctype, body = home
    best_url, best = url, _parse(ctype, body)
    links = find_menu_links(body, url) if ctype != "application/pdf" else []
    for link in links:
        menu = _read_menu(fetcher, link)
        if menu is not None and _is_better(menu, best):
            best_url, best = link, menu

    if not links and len(best["items"]) < _MIN_ITEMS:
        out["status"] = "NO_MENU_LINK"
        return out
    out.update(menu_url=best_url, items=best["items"], n_items=len(best["items"]))
    if out["n_items"] >= _MIN_ITEMS:
        out["status"] = "MENU_PDF" if best["format"] == "PDF" else "MENU_TEXT"
    elif best["n_images"] >= _MIN_IMAGES:
        out["status"] = "MENU_IMAGES"
    else:
        out["status"] = "MENU_NO_PRICES"
    return out
