"""Метрики качества кластеризации (требование задачи):
SW — silhouette width, CH — Calinski-Harabasz, S_Dbw — Davies-Bouldin,
AVI — variation of information vs официальный перечень, AVU — adjusted Rand index,
MQ — modularity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.metrics.cluster import calinski_harabasz_score


def vi(x: np.ndarray, y: np.ndarray) -> float:
    """Variation of information: H(x) + H(y) - 2*I(x;y), биты."""
    n = len(x)

    def ent(v: np.ndarray) -> float:
        p = np.bincount(v) / n
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    j = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(j, (xi, yi), 1)
    p = j / n
    # I = sum p(x,y) log( p(x,y) / (p(x)p(y)) )
    px = np.bincount(xi) / n
    py = np.bincount(yi) / n
    mask = p > 0
    marg = px[:, None] * py[None, :]
    i_val = float(-(p[mask] * np.log2(p[mask] / marg[mask])).sum())
    return ent(np.asarray(xi)) + ent(np.asarray(yi)) - 2 * i_val

from .features import FEAT_COLS, load_features
from .network.graphs import candidate_pairs, louvain, window_edges


def main() -> None:
    feats, _ = load_features()
    ids = np.sort(feats.mo.unique())
    clusters = pd.read_csv("out/clusters_all.csv")
    final = clusters[clusters.window == clusters.window.max()]
    member = final.set_index("mo").cluster.reindex(ids).to_numpy()

    w = feats[feats.t >= 18]
    X = w.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0).to_numpy(float)

    # только кластеры >= 3 МО (исключаем шум)
    keep = np.array([ (member == c).sum() >= 3 for c in np.unique(member) ])
    big = np.isin(member, np.unique(member)[keep])
    Xk, yk = X[big], member[big]

    metrics = {
        "SW (silhouette)": round(float(silhouette_score(Xk, yk)), 3),
        "CH (calinski-harabasz)": round(float(calinski_harabasz_score(Xk, yk)), 1),
        "S_Dbw (davies-bouldin)": round(float(davies_bouldin_score(Xk, yk)), 3),
    }

    # сравнение с официальным перечнем (общие МО)
    off = pd.read_csv("data/sources/official_agglomerations.csv")
    off = off.rename(columns={"agg_name": "Агломерация"})
    off = off[off.territory_id.notna()]
    ref = {int(t): a for t, a in zip(off.territory_id, off["Агломерация"], strict=True)}
    common = np.array([int(t) in ref for t in ids[big]])
    y_ref = np.array([ref[int(t)] for t in ids[big][common]])
    y_mar = yk[common]
    metrics["AVI (VI vs official, nat.)"] = round(float(vi(np.asarray(y_mar), np.asarray(y_ref))), 3)
    metrics["AVU (ARI vs official)"] = round(float(adjusted_rand_score(y_mar, y_ref)), 3)

    # modularity — из пайплайна
    pairs, _ = candidate_pairs("data/sources", ids)
    edges = window_edges(feats, 23, pairs, ids)
    _, q = louvain(len(ids), edges)
    metrics["MQ (modularity)"] = round(q, 3)

    row = pd.DataFrame(list(metrics.items()), columns=["metric", "value"]).set_index("metric")
    row.to_csv("out/metrics.csv")
    print(row.T.to_string())
    print(f"\nМО в анализе (кластеры >=3): {int(big.sum())} из {len(ids)}")


if __name__ == "__main__":
    main()
