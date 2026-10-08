"""EXP 3: сравнение методов регионализации (Lattimer 2022, обзор).

Методы на одних данных (14-осевая матрица X):
  (а) наш метод: mutual-kNN (top-8) по косинусу 14 осей + Louvain
      (без ограничения связности);
  (б) Ward-агломерация с ограничением связности;
  (в) SKATER: MST по графу связности (вес = 1 - косинус), резка K-1 рёбер.

Связность для (б),(в): триангуляция Делоне по центроидам МО (~6 тыс. рёбер,
граф связен) — стандарт для регионализации (аналог AZP/REDCAP). Почему не
кандидаты <=50/200 км: при 50 км граф распадается на 856 компонент, при
200 км — 98 (Арктика/Дальневосток) — связность на масштабе агломерации
слишком жёсткая для макрорегионализации.

Все три дают K ~ 17 регионов. Метрики: пространственная когерентность
(+20 перестановок, p), ARI vs официальный перечень, силуэт по X,
Polsby–Popper регионов (объединение контуров МО).

Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.paper_exp3_regionalization.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay

from .papers_common import (coords, load_ctx, neighbors, spatial_coh,
                                   typ14)
from .features_exp import ari_all

K_TARGET = 17


def delaunay_pairs(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    # Делоне в метрической проекции (6933), иначе в градусах Дальневосток
    # «склеивается» по долготе
    import pyproj
    t = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    # always_xy: первый аргумент = x = lon
    p = np.column_stack(t.transform(lon, lat))
    # микроджиттер 1–10 м: ломает cocircularity (кольца внутригородских
    # округов) и точные дубликаты; для графа связности неважно
    p += np.random.default_rng(0).uniform(1.0, 10.0, size=p.shape)
    tri = Delaunay(p)
    e = set()
    for t in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            e.add((min(int(t[a]), int(t[b])), max(int(t[a]), int(t[b]))))
    return np.array(sorted(e), dtype=np.int64)


def ward_connected(X: np.ndarray, conn: csr_matrix, k_total: int) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering
    m = AgglomerativeClustering(n_clusters=k_total, linkage="ward",
                                connectivity=conn)
    return m.fit_predict(X)


def skater(sim: np.ndarray, pairs: np.ndarray, k_total: int) -> np.ndarray:
    """MST на графе связности, вес = 1 - cos; вырезаем K-1 самых тяжёлых рёбер."""
    n = len(sim)
    ii, jj = pairs[:, 0], pairs[:, 1]
    w = 1.0 - sim[ii, jj]
    A = csr_matrix((np.r_[w, w], (np.r_[ii, jj], np.r_[jj, ii])), shape=(n, n))
    mst = minimum_spanning_tree(A).tocoo()
    keep = np.argsort(mst.data)[: len(mst.data) - (k_total - 1)]
    Ak = csr_matrix((np.ones(len(keep)), (mst.row[keep], mst.col[keep])),
                    shape=(n, n))
    from scipy.sparse.csgraph import connected_components
    _, ck = connected_components(Ak, directed=False)
    return ck


def pp(geom) -> float:
    import pyproj
    from shapely.ops import transform
    lon = geom.centroid.x
    zone = int((lon + 180) / 6) % 60 + 1
    t = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:326{zone:02d}", always_xy=True)
    g = transform(t.transform, geom)
    a, p = g.area, g.length
    return 4 * np.pi * a / (p * p) if p > 0 else 0.0


def region_pp(geoms_by_id: dict, ids: np.ndarray, members: np.ndarray) -> float:
    from shapely.ops import unary_union
    by_t: dict[int, list] = {}
    for i, tid in enumerate(ids):
        by_t.setdefault(int(members[i]), []).append(geoms_by_id[tid])
    return float(np.median([pp(unary_union(g)) for g in by_t.values()]))


def main() -> None:
    import geopandas as gpd

    feats, ids, pairs, agg_by_id = load_ctx()
    lat, lon = coords(ids)
    X, cols, mt, q = typ14(feats, ids, pairs)
    norm = np.linalg.norm(X, axis=1).clip(min=1e-9)
    sim = (X / norm[:, None]) @ (X / norm[:, None]).T
    np.fill_diagonal(sim, 1.0)

    pd_del = delaunay_pairs(lat, lon)
    conn = csr_matrix((np.ones(2 * len(pd_del)),
                       (np.r_[pd_del[:, 0], pd_del[:, 1]],
                        np.r_[pd_del[:, 1], pd_del[:, 0]])),
                      shape=(len(ids), len(ids)))

    g = gpd.read_file("data/results/geojson/mo.geojson", columns=["territory_id"])
    geo_by_id = dict(zip(g.territory_id.to_numpy(), g.geometry.make_valid()))
    nb = neighbors(pairs)  # когерентность — на исходных 50 км

    def row(method: str, members: np.ndarray) -> dict:
        coh, null_max, p = spatial_coh(members, nb)
        from sklearn.metrics import silhouette_score
        sw = float(silhouette_score(X, members))
        return {"method": method, "K": len(np.unique(members)),
                "spatial_coh": round(coh, 3), "null_max": round(null_max, 3),
                "p": round(p, 3), "ARI": round(ari_all(members, ids, agg_by_id), 3),
                "silhouette": round(sw, 3),
                "PP_med": round(region_pp(geo_by_id, ids, members), 4)}

    rows = [row("mutual-kNN + Louvain (наш)", mt)]
    rows.append(row("Ward + connectivity (K=17)",
                    ward_connected(X, conn, K_TARGET)))
    rows.append(row("SKATER (MST, K=17)", skater(sim, pd_del, K_TARGET)))

    out = pd.DataFrame(rows)
    out.to_csv("out/paper_exp3_regionalization.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
