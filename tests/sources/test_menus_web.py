import pytest

from puzzle_recsys.sources.menus_web import (
    BLOCKED_DOMAINS,
    extract_price_lines,
    find_menu_links,
    normalize_website,
    pilot_restaurant,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("WWW.SALVAJE.ART", "https://www.salvaje.art"),
        ("www.maison-kayser.com .mx", "https://www.maison-kayser.com.mx"),
        ("https://foo.mx/menu", "https://foo.mx/menu"),
        ("www.salvaje_mx", None),  # el caso real de Salvaje en el DENUE
        ("www.bolichera.21", None),
        ("", None),
        (None, None),
        ("www.facebook.com/turestaurante", None),  # redes sociales: fuera de alcance
    ],
)
def test_normalize_website(raw, expected) -> None:
    assert normalize_website(raw) == expected


def test_find_menu_links_keeps_menu_and_pdf_links_on_same_site() -> None:
    html = """
    <a href="/menu/">Menú</a>
    <a href="https://salvaje.art/desayunos/">Desayunos</a>
    <a href="/files/carta-2025.pdf">Descarga la carta</a>
    <a href="/contacto">Contacto</a>
    <a href="https://www.instagram.com/salvaje">IG</a>
    """
    links = find_menu_links(html, "https://salvaje.art/")
    assert "https://salvaje.art/menu/" in links
    assert "https://salvaje.art/desayunos/" in links
    assert "https://salvaje.art/files/carta-2025.pdf" in links
    assert all("contacto" not in u and "instagram" not in u for u in links)


def test_extract_price_lines() -> None:
    text = """ENTRADAS
    Kushiages 3 piezas de queso 140
    Edamames $ 140
    Ramen Pork belly, yakisoba $240.00
    Horario 9 AM - 10 PM
    Tel 5573619858
    """
    items = extract_price_lines(text)
    assert ("Kushiages 3 piezas de queso", 140.0) in items
    assert ("Edamames", 140.0) in items
    assert ("Ramen Pork belly, yakisoba", 240.0) in items
    assert all(name not in {"Horario 9 AM - 10 PM", "Tel"} for name, _ in items)


class FakeFetcher:
    """Simula la web: url → (content_type, cuerpo). Registra lo que se pidió."""

    def __init__(self, pages, disallowed=()):
        self.pages = pages
        self.disallowed = set(disallowed)
        self.requested = []

    def allowed(self, url):
        return url not in self.disallowed

    def get(self, url):
        self.requested.append(url)
        return self.pages.get(url)


def test_pilot_reads_text_menu() -> None:
    menu = "\n".join(f"Platillo {i} 1{i}0" for i in range(1, 9))
    fetcher = FakeFetcher(
        {
            "https://x.mx": ("text/html", '<a href="/menu">Menú</a>'),
            "https://x.mx/menu": ("text/html", f"<div>{menu}</div>"),
        }
    )
    out = pilot_restaurant(fetcher, "r1", "x.mx")
    assert out["status"] == "MENU_TEXT"
    assert out["n_items"] == 8


def test_pilot_detects_image_menu() -> None:
    imgs = "".join(f'<img src="/wp-content/uploads/menu{i}.png">' for i in range(6))
    fetcher = FakeFetcher(
        {
            "https://x.mx": ("text/html", '<a href="/menu">Menú</a>'),
            "https://x.mx/menu": ("text/html", f"<p>Horario</p>{imgs}"),
        }
    )
    assert pilot_restaurant(fetcher, "r1", "x.mx")["status"] == "MENU_IMAGES"


def test_pilot_respects_robots() -> None:
    fetcher = FakeFetcher({"https://x.mx": ("text/html", "")}, disallowed={"https://x.mx"})
    out = pilot_restaurant(fetcher, "r1", "x.mx")
    assert out["status"] == "ROBOTS_DISALLOW"
    assert fetcher.requested == []  # no se pidió NADA


def test_pilot_handles_invalid_and_unreachable() -> None:
    assert pilot_restaurant(FakeFetcher({}), "r1", "www.salvaje_mx")["status"] == "INVALID_URL"
    assert pilot_restaurant(FakeFetcher({}), "r1", "x.mx")["status"] == "UNREACHABLE"


def test_blocked_domains_include_platforms_whose_terms_forbid_scraping() -> None:
    assert {"google.com", "instagram.com", "facebook.com"} <= set(BLOCKED_DOMAINS)


# ---------------------------------------------- regresiones (fallas del primer piloto real)


def test_links_with_spaces_are_percent_encoded() -> None:
    html = '<a href="/docs/Menu Comida 2025.pdf">Menú</a>'
    assert find_menu_links(html, "https://x.mx/") == ["https://x.mx/docs/Menu%20Comida%202025.pdf"]


def test_legal_documents_are_not_menus() -> None:
    html = """
    <a href="/assets/web/docs/2023 03 31 Aviso Legal Es 100M.pdf">Aviso legal</a>
    <a href="/privacidad.pdf">Aviso de privacidad</a>
    <a href="/terminos">Términos y condiciones</a>
    """
    assert find_menu_links(html, "https://x.mx/") == []


def test_pilot_never_raises_even_if_the_fetcher_explodes() -> None:
    import http.client

    class Exploding(FakeFetcher):
        def get(self, url):
            raise http.client.InvalidURL("URL can't contain control characters")

    out = pilot_restaurant(Exploding({}), "r1", "x.mx")
    assert out["status"] == "UNREACHABLE"
