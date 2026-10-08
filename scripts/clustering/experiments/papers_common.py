"""Общий контекст для paper-экспериментов (статьи в /sources).

Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.paper_expN_*.py
Все эксперименты читают только существующие данные/артефакты, производство не трогают.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features_exp import STATIC, louvain, zscore  # noqa: F401 (re-export)
from .features_exp4 import basket_coef_by_mo
from ..features import FEAT_COLS, load_features
from ..network.graphs import KNN, candidate_pairs

STAT14 = STATIC + ["log_basket", "growth_basket"]


def load_ctx() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """feats, ids, пары <= 50 км, словарь МО->официальная агломерация."""
    feats, _ = load_features()
    ids = np.sort(feats.mo.unique())
    pairs, _ = candidate_pairs("data/sources", ids)
    off = pd.read_csv("data/experiments/inputs/derived/official_agglomerations_mapped.csv")
    off = off[off.territory_id.notna()]
    agg_by_id = dict(zip(off.territory_id, off["Агломерация"]))
    return feats, ids, pairs, agg_by_id


def coords(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Центроиды МО (lat, lon), порядок = ids. Из mo.geojson (все 2548 МО)."""
    import geopandas as gpd
    g = gpd.read_file("data/results/geojson/mo.geojson", columns=["territory_id"])
    c = g.geometry.to_crs("EPSG:4326").centroid
    p = pd.DataFrame({"lat": c.y, "lon": c.x}).set_index(g.territory_id.to_numpy())
    p = p.reindex(ids)
    assert p.notna().all().all(), f"нет координат: {int(p.isna().any(axis=1).sum())}"  # type: ignore[attr-defined]
    return p.lat.to_numpy(float), p.lon.to_numpy(float)


def haversine(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Матрица расстояний, км (плотная)."""
    R = 6371.0
    la1, lo1 = np.deg2rad(lat)[:, None], np.deg2rad(lon)[:, None]
    la2, lo2 = np.deg2rad(lat)[None, :], np.deg2rad(lon)[None, :]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.clip(np.sqrt(h), 0.0, 1.0))


def population(ids: np.ndarray) -> np.ndarray:
    """Население 2024 (масса для гравитационной модели), порядок = ids."""
    pop = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                      usecols=["territory_id", "pop_2023", "pop_2024"])
    p = pop.set_index("territory_id").reindex(ids)
    v = p.pop_2024.fillna(p.pop_2023).fillna(float(p.pop_2024.median()))
    return v.to_numpy(float)


def prod_members(ids: np.ndarray) -> np.ndarray:
    """Размещение производства (финальное окно 12/2024)."""
    cl = pd.read_csv("out/clusters_all.csv")
    final = cl[cl.window == cl.window.max()]
    return final.set_index("mo").cluster.reindex(ids).to_numpy()


def prod_edges() -> np.ndarray:
    """(m,3): i, j (индексы в ids), w — рёбра производства 12/2024."""
    e = pd.read_csv("out/edges_2024-12.csv")
    return e.to_numpy(float)


def static14(ids: np.ndarray) -> pd.DataFrame:
    """6 статических + log_basket + growth_basket, index=ids."""
    static = pd.read_csv("data/experiments/inputs/derived/mo_static_features.csv").set_index("territory_id")
    stat = static[STATIC].reindex(ids).fillna(0.0)
    b = basket_coef_by_mo().reindex(ids).to_numpy(float)
    b = np.where(np.isnan(b), np.nanmedian(b, axis=1, keepdims=True), b)
    stat["log_basket"] = np.log1p(b.mean(axis=1))
    stat["growth_basket"] = b[:, 12:24].mean(1) / np.clip(b[:, 0:12].mean(1), 1e-9, None) - 1
    return stat


def typ14(feats: pd.DataFrame, ids: np.ndarray, pairs: np.ndarray):
    """Типология 14 осей: X (z), cols, members (mutual kNN + Louvain), q."""
    dyn = feats.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0)
    stat = static14(ids)
    X = zscore(np.hstack([dyn.to_numpy(float), stat[STAT14].to_numpy(float)]))
    cols = FEAT_COLS + STAT14
    norm = np.linalg.norm(X, axis=1).clip(min=1e-9)
    sim = (X / norm[:, None]) @ (X / norm[:, None]).T
    np.fill_diagonal(sim, -1.0)
    sim = np.where(sim > 0, sim, -1.0)
    idx = np.argsort(-sim, axis=1)[:, :KNN]
    iu = np.repeat(np.arange(len(ids)), KNN)
    ju = idx.ravel()
    w = sim[iu, ju]
    ok = w > 0
    lo, hi = np.minimum(iu[ok], ju[ok]), np.maximum(iu[ok], ju[ok])
    uniq = dict(zip(zip(lo, hi), w[ok], strict=True))
    edges = [(int(i), int(j), float(w)) for (i, j), w in uniq.items()]
    members, q = louvain(len(ids), edges)
    return X, cols, members, q


def neighbors(pairs: np.ndarray) -> dict[int, set[int]]:
    nb: dict[int, set[int]] = {}
    for a, b in zip(*pairs.T):
        nb.setdefault(int(a), set()).add(int(b))
        nb.setdefault(int(b), set()).add(int(a))
    return nb


def spatial_coh(members: np.ndarray, nb: dict[int, set[int]], seed: int = 42,
                n_null: int = 20) -> tuple[float, float, float]:
    """(coherence, null_max, p) — доля МО с соседом своего класса <=50 км."""
    def coh(m: np.ndarray) -> float:
        return sum(any(m[j] == m[i] for j in nb.get(i, ())) for i in range(len(m))) / len(m)

    c = coh(members)
    rng = np.random.default_rng(seed)
    sh = [coh(rng.permutation(members)) for _ in range(n_null)]
    p = (sum(s >= c for s in sh) + 1) / (n_null + 1)
    return c, max(sh), p
