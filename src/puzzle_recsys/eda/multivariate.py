"""Análisis multivariado: correlaciones, MCA y clustering de comensales.

Tres preguntas que el bivariado no contesta:

1. **Correlaciones en bloque:** qué features del comensal dicen lo mismo
   (redundancia que infla el modelo sin aportar).
2. **MCA** (análisis de correspondencias múltiples): la estructura conjunta de
   las variables **categóricas** de la mesa (franja, cocina, tamaño, nivel de
   precio). Es el equivalente categórico del PCA.
3. **Clustering de comensales:** si el comportamiento observado forma grupos, y
   si esos grupos **recuperan los gustos latentes** de la simulación. La verdad
   oculta se usa solo para verificar, nunca como entrada.

Usado en: notebooks 08.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import prince
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from puzzle_recsys.eda.bivariate import effect_label, kruskal_effect

COURSES = ("MAIN", "DRINK", "STARTER", "SIDE", "DESSERT")


def diner_behavior_matrix(
    sessions: pd.DataFrame,
    guests: pd.DataFrame,
    order_items: pd.DataFrame,
    menu_items: pd.DataFrame,
    *,
    min_visits: int = 5,
) -> pd.DataFrame:
    """Una fila por comensal registrado con su comportamiento **observado**.

    ``sessions`` necesita ``session_id``, ``restaurant_id``, ``cocina`` y
    ``ticket_por_persona`` (la salida de ``session_frame`` sirve). Columnas:

    - ``cocina__<c>``: proporción de sus visitas a cada cocina.
    - ``por_visita__<plato>``: líneas propias e individuales por visita, por tipo
      de plato (los hábitos: ¿pide entrada? ¿postre?).
    - ``picante_pedido``: picante medio de sus líneas propias e individuales.
    - ``log_ticket_pp``: log del ticket por persona mediano (cola larga).
    - ``tasa_compartir``: proporción de sus mesas con al menos una línea compartida.
    - ``variedad``: restaurantes distintos / visitas (en vez de las dos por
      separado, que tienen rho = 0.84 en el notebook 07).
    - ``log_visitas``: frecuencia.

    Se descartan los comensales con menos de ``min_visits`` visitas: con 2 o 3
    visitas las proporciones son ruido.
    """
    seated = guests.dropna(subset=["diner_id"]).merge(
        sessions[["session_id", "restaurant_id", "cocina", "ticket_por_persona"]], on="session_id"
    )
    by = seated.groupby("diner_id")
    visits = by.size()
    keep = visits.index[visits >= min_visits]
    seated = seated[seated["diner_id"].isin(keep)]
    by = seated.groupby("diner_id")
    visits = by.size()

    shares = pd.crosstab(seated["diner_id"], seated["cocina"], normalize="index")
    shares.columns = [f"cocina__{c}" for c in shares.columns]

    own = order_items.loc[~order_items["is_shared"], ["guest_id", "menu_item_id"]]
    own = own.merge(seated[["guest_id", "diner_id"]], on="guest_id")
    own = own.merge(menu_items[["menu_item_id", "course", "spice_level"]], on="menu_item_id")
    per_visit = (
        pd.crosstab(own["diner_id"], own["course"])
        .reindex(index=visits.index, columns=list(COURSES), fill_value=0)
        .div(visits, axis=0)
    )
    per_visit.columns = [f"por_visita__{c}" for c in per_visit.columns]

    shared_sessions = set(order_items.loc[order_items["is_shared"], "session_id"])
    out = pd.concat([shares, per_visit], axis=1)
    out["picante_pedido"] = own.groupby("diner_id")["spice_level"].mean()
    out["log_ticket_pp"] = np.log(by["ticket_por_persona"].median())
    out["tasa_compartir"] = (
        seated["session_id"].isin(shared_sessions).groupby(seated["diner_id"]).mean()
    )
    out["variedad"] = by["restaurant_id"].nunique() / visits
    out["log_visitas"] = np.log(visits)
    return out.rename_axis("diner_id")


def redundant_pairs(corr: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Pares de variables con ``|rho| >= threshold`` en una matriz de correlación.

    Cada par aparece una vez (triángulo superior), ordenado por ``|rho|``.
    """
    cols = list(corr.columns)
    rows = [
        {"a": a, "b": b, "rho": float(corr.loc[a, b])}
        for i, a in enumerate(cols)
        for b in cols[i + 1 :]
        if abs(corr.loc[a, b]) >= threshold
    ]
    out = pd.DataFrame(rows, columns=["a", "b", "rho"])
    return out.reindex(out["rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def table_categories(mesas: pd.DataFrame, *, top_cuisines: int = 8) -> pd.DataFrame:
    """Variables categóricas de la mesa para el MCA.

    Las cocinas fuera de las ``top_cuisines`` más comunes se agrupan en "otras"
    (una categoría con muy pocas mesas domina el MCA sin decir nada), y los números
    se agrupan: personas en 1 / 2 / 3-4 / 5+ y nivel de precio en cuartiles.
    """
    top = mesas["cocina"].value_counts().index[:top_cuisines]
    return pd.DataFrame(
        {
            "franja": mesas["franja"].astype(str),
            "fin_de_semana": np.where(mesas["fin_de_semana"], "fin de semana", "entre semana"),
            "cocina": mesas["cocina"].where(mesas["cocina"].isin(top), "otras"),
            "personas": pd.cut(mesas["pax"], [0, 1, 2, 4, np.inf], labels=["1", "2", "3-4", "5+"]),
            "nivel_precio": pd.qcut(mesas["nivel_precio"], 4, labels=["Q1", "Q2", "Q3", "Q4"]),
            "personal": mesas["personal"].astype(str),
        },
        index=mesas.index,
    )


def benzecri_inertia(eigenvalues: np.ndarray, n_variables: int) -> np.ndarray:
    """Porcentaje de inercia de cada dimensión del MCA con la corrección de Benzécri.

    El MCA "crudo" subestima mucho la importancia de las primeras dimensiones,
    porque la codificación one-hot fabrica inercia artificial. Benzécri (1979)
    se queda solo con los eigenvalores mayores que el promedio ``1/K`` y los
    reescala: ``((K / (K - 1)) * (lambda - 1/K))**2``.
    """
    k = n_variables
    lam = np.asarray(eigenvalues, dtype=float)
    corrected = np.where(lam > 1 / k, ((k / (k - 1)) * (lam - 1 / k)) ** 2, 0.0)
    return 100 * corrected / corrected.sum()


def mca_summary(df: pd.DataFrame, *, n_components: int = 5, seed: int = 0) -> dict:
    """Ajusta un MCA sobre columnas categóricas y resume inercia y categorías.

    Returns
    -------
    dict
        ``inercia``: por dimensión, eigenvalor, % crudo y % corregido (Benzécri).
        ``categorias``: ``variable``, ``categoria``, coordenadas ``dim1..`` y la
        contribución de cada categoría a las dos primeras dimensiones.
    """
    data = df.astype(str)
    mca = prince.MCA(n_components=n_components, random_state=seed).fit(data)
    eig = np.asarray(mca.eigenvalues_)
    inertia = pd.DataFrame(
        {
            "dimension": np.arange(1, len(eig) + 1),
            "eigenvalor": eig,
            "pct_crudo": np.asarray(mca.percentage_of_variance_),
            "pct_benzecri": benzecri_inertia(eig, data.shape[1]),
        }
    )
    coords = mca.column_coordinates(data)
    coords.columns = [f"dim{j + 1}" for j in range(coords.shape[1])]
    contrib = mca.column_contributions_.iloc[:, :2].sum(axis=1)
    var_cat = coords.index.to_series().str.split("__", n=1, expand=True)
    cats = pd.concat(
        [
            var_cat.set_axis(["variable", "categoria"], axis=1),
            coords,
            contrib.rename("contribucion_1_2"),
        ],
        axis=1,
    ).reset_index(drop=True)
    return {"inercia": inertia, "categorias": cats}


def kmeans_scan(
    x: np.ndarray,
    ks: Iterable[int],
    *,
    seed: int = 0,
    silhouette_sample: int | None = None,
) -> pd.DataFrame:
    """K-means para cada ``k``: inercia (codo) y silueta (cohesión vs separación).

    La silueta cuesta O(n²); con muchos comensales se calcula sobre una muestra.
    """
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
        sil = silhouette_score(x, km.labels_, sample_size=silhouette_sample, random_state=seed)
        rows.append({"k": k, "inercia": float(km.inertia_), "silueta": float(sil)})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ClusterResult:
    """Resultado de :func:`cluster_diners`."""

    labels: pd.Series
    n_components: int
    explained: float
    embedding: np.ndarray
    centers: pd.DataFrame


def cluster_diners(
    features: pd.DataFrame, *, k: int, seed: int = 0, variance: float = 0.9
) -> ClusterResult:
    """Estandariza, reduce con PCA y agrupa con K-means.

    El PCA se queda con los componentes que explican ``variance`` de la varianza:
    quita ruido y la redundancia que encontró la matriz de correlación. Los
    ``centers`` se reportan en las **unidades originales** (media por grupo), que
    es lo que se puede interpretar.
    """
    z = StandardScaler().fit_transform(features.to_numpy())
    pca = PCA(n_components=variance, svd_solver="full", random_state=seed).fit(z)
    emb = pca.transform(z)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb)
    labels = pd.Series(km.labels_, index=features.index, name="cluster")
    return ClusterResult(
        labels=labels,
        n_components=int(pca.n_components_),
        explained=float(pca.explained_variance_ratio_.sum()),
        embedding=emb,
        centers=features.groupby(labels).mean(),
    )


def recovery_vs_truth(
    labels: pd.Series,
    numeric_truth: pd.DataFrame,
    categorical_truth: Mapping[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Cuánto de cada rasgo **latente** recuperan los grupos observados.

    - Rasgos numéricos: ε² de Kruskal-Wallis del rasgo según el grupo.
    - Rasgos categóricos: información mutua ajustada (AMI; 0 = azar, 1 = idéntico).

    Todo se alinea por índice con ``labels``.
    """
    rows = []
    for col in numeric_truth.columns:
        frame = pd.DataFrame({"v": numeric_truth[col], "g": labels}).dropna()
        res = kruskal_effect(frame, "v", "g")
        rows.append(
            {"rasgo": col, "metrica": "epsilon2", "valor": res["epsilon2"], "efecto": res["efecto"]}
        )
    for name, truth in (categorical_truth or {}).items():
        frame = pd.DataFrame({"t": truth, "g": labels}).dropna()
        ami = float(adjusted_mutual_info_score(frame["t"], frame["g"]))
        rows.append(
            {"rasgo": name, "metrica": "AMI", "valor": ami, "efecto": effect_label(ami, "V")}
        )
    return pd.DataFrame(rows)


def trait_recovery(
    observed: pd.DataFrame, latent: pd.DataFrame, pairs: Iterable[tuple[str, str]]
) -> pd.DataFrame:
    """Spearman entre cada feature **observada** y el rasgo **latente** que debería reflejar.

    Es la prueba de que el comportamiento contiene la señal que el modelo tiene
    que aprender (la verdad oculta solo se usa para verificar).
    """
    rows = []
    for obs, lat in pairs:
        pair = pd.DataFrame({"o": observed[obs], "l": latent[lat]}).dropna()
        rho = float(stats.spearmanr(pair["o"], pair["l"])[0])
        rows.append(
            {"observado": obs, "latente": lat, "rho": rho, "efecto": effect_label(rho, "rho")}
        )
    return pd.DataFrame(rows)


def sharing_by_table_trait(
    sessions: pd.DataFrame,
    guest_traits: pd.DataFrame,
    order_items: pd.DataFrame,
    *,
    q: int = 3,
) -> pd.DataFrame:
    """Qué proporción de mesas comparte algo, por cuantil del gusto medio por compartir.

    ``guest_traits`` trae ``session_id`` y ``trait`` (el rasgo latente de cada
    persona sentada). Solo cuentan las mesas de 2 o más personas.
    """
    trait = guest_traits.groupby("session_id")["trait"].mean().rename("trait")
    s = sessions.loc[sessions["pax"] >= 2, ["session_id"]].join(trait, on="session_id")
    s = s.dropna(subset=["trait"])
    shared = set(order_items.loc[order_items["is_shared"], "session_id"])
    s["comparte"] = s["session_id"].isin(shared)
    s["cuantil"] = pd.qcut(s["trait"].rank(method="first"), q, labels=range(1, q + 1))
    by = s.groupby("cuantil", observed=True)
    return pd.DataFrame(
        {
            "mesas": by.size(),
            "gusto_medio": by["trait"].mean(),
            "pct_comparten": 100 * by["comparte"].mean(),
        }
    ).reset_index()


def latent_diner_traits(
    tastes: pd.DataFrame,
    duplicates: pd.DataFrame,
    diner_ids: pd.Index,
    cuisines: Sequence[str],
) -> pd.DataFrame:
    """Gustos latentes (verdad oculta) de cada comensal observado, solo para verificar.

    Una cuenta duplicada hereda los gustos de la persona real que la abrió.
    Agrega ``cocina_dominante``: la cocina con mayor afinidad latente.
    """
    true_of = duplicates.set_index("diner_id")["diner_id_true"]
    ids = diner_ids.to_series()
    out = tastes.set_index("diner_id").loc[ids.map(true_of).fillna(ids)].set_axis(diner_ids)
    affinity = np.vstack(out["cuisine_affinity"].to_numpy())
    out["cocina_dominante"] = np.asarray(cuisines)[affinity.argmax(axis=1)]
    return out


def guest_trait(
    guests: pd.DataFrame,
    taste_refs: pd.DataFrame,
    taste_tables: Sequence[pd.DataFrame],
    trait: str,
) -> pd.DataFrame:
    """Rasgo latente de **cada persona sentada**, registrada o acompañante anónimo.

    ``taste_refs`` (``guest_tastes`` de la verdad oculta) apunta cada invitado a una
    fila de alguna de las ``taste_tables`` (``diner_tastes``, ``anonymous_tastes``).
    """
    lookup = pd.concat([t.set_index("diner_id")[trait] for t in taste_tables])
    out = guests[["guest_id", "session_id"]].merge(taste_refs, on="guest_id")
    out["trait"] = out["taste_ref"].map(lookup).to_numpy()
    return out[["guest_id", "session_id", "trait"]]
