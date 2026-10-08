"""Подбор KNN и resolution по чистоте кластеров относительно официального перечня."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import load_features
from .network.graphs import candidate_pairs, window_edges


def purity(members: np.ndarray, ids: np.ndarray, agg_by_id: dict) -> float:
    """Доля МО, попавших в кластер, где их официальная агломерация доминирует."""
    tot, hit = 0, 0
    for cid in np.unique(members):
        mem = members == cid
        tids = ids[mem]
        counts: dict[str, int] = {}
        for t in tids:
            a = agg_by_id.get(int(t))
            if a:
                counts[a] = counts.get(a, 0) + 1
        if not counts:
            continue
        dom = max(counts.values())
        tot += sum(counts.values())
        hit += dom
    return hit / tot if tot else 0.0


def main() -> None:
    feats, months = load_features()
    ids = np.sort(feats.mo.unique())
    off = pd.read_csv("data/experiments/inputs/derived/official_agglomerations_mapped.csv")
    off = off[off.territory_id.notna()]
    agg_by_id = dict(zip(off.territory_id, off["Агломерация"], strict=True))
    pairs, _ = candidate_pairs("data/sources", ids)
    t_end = 23  # финальное окно

    rows: list[dict] = []
    for knn in (8, 15, 25):
        for res in (1.0, 1.5, 2.0):
            edges = window_edges(feats, t_end, pairs, ids, k=knn)
            member, q = louvain_res(len(ids), edges, res)
            n3 = sum(1 for c in np.unique(member) if (member == c).sum() >= 3)
            rows.append({"knn": knn, "res": res, "clusters": len(np.unique(member)),
                         "clusters_3plus": n3, "q": round(q, 3),
                         "purity": round(purity(member, ids, agg_by_id), 3)})
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))
    grid.to_csv("out/tune_grid.csv", index=False)


def louvain_res(n: int, edges, resolution: float):
    import igraph as ig
    g = ig.Graph(n, [(e[0], e[1]) for e in edges])
    g.es["weight"] = [e[2] for e in edges]
    cl = g.community_multilevel(weights="weight", resolution=resolution)
    return np.asarray(cl.membership), float(cl.q)


if __name__ == "__main__":
    main()
