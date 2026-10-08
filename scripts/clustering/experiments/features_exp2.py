"""Быстрые доп-эксперименты поверх features_exp: окна 12/24 мес, K=15, гибрид весов."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features_exp import STATIC, ari_all, run_variant, window_edges_cols, zscore
from ..features import FEAT_COLS, load_features
from ..network.graphs import KNN, candidate_pairs, louvain, window_edges
from ..tune import purity

T_END = 23


def weighted_edges(feats12: pd.DataFrame, t_end: int, pairs: np.ndarray, ids: np.ndarray,
                   w_dyn: float, window: int, k: int = KNN) -> list[tuple[int, int, float]]:
    """w_dyn * cos_z(6 dyn) + (1-w_dyn) * cos_z(6 stat)."""
    w = feats12[(feats12.t >= t_end - window + 1) & (feats12.t <= t_end)]
    dyn = zscore(w.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0).to_numpy(float))
    stat = zscore(w.groupby("mo")[STATIC].mean().reindex(ids).fillna(0.0).to_numpy(float))
    for M in (dyn, stat):
        n = np.linalg.norm(M, axis=1).clip(min=1e-9)
        M = M / n[:, None]
    a, b = pairs[:, 0], pairs[:, 1]
    cos = w_dyn * (dyn[a] * dyn[b]).sum(1) + (1 - w_dyn) * (stat[a] * stat[b]).sum(1)
    cos = np.where(cos > 0, cos, np.nan)
    df = pd.DataFrame({"i": a, "j": b, "w": cos}).dropna(subset=["w"])
    both = pd.concat([df, df.assign(i=df.j, j=df.i)], ignore_index=True)
    both = both.sort_values("w", ascending=False).groupby("i").head(k)
    lo, hi = np.minimum(both.i, both.j), np.maximum(both.i, both.j)
    uniq = dict(zip(zip(lo, hi), both.w.to_numpy(), strict=True))
    return [(int(i), int(j), float(w)) for (i, j), w in uniq.items()]


def main() -> None:
    feats, _ = load_features()
    ids = np.sort(feats.mo.unique())
    off = pd.read_csv("data/experiments/inputs/derived/official_agglomerations_mapped.csv")
    off = off[off.territory_id.notna()]  # type: ignore[truthy-function]
    agg_by_id = dict(zip(off.territory_id, off["Агломерация"], strict=True))
    pairs, _ = candidate_pairs("data/sources", ids)
    static = pd.read_csv("data/experiments/inputs/derived/mo_static_features.csv").set_index("territory_id")
    feats12 = feats.merge(static.reset_index().rename(columns={"territory_id": "mo"}),
                          on="mo", how="left")

    rows = [
        run_variant("prod w12", feats, T_END, pairs, ids, FEAT_COLS, z=False,
                    agg_by_id=agg_by_id),
        run_variant("prod w24", feats, T_END, pairs, ids, FEAT_COLS, z=False,
                    agg_by_id=agg_by_id),
        run_variant("prod K15", feats, T_END, pairs, ids, FEAT_COLS, z=False,
                    agg_by_id=agg_by_id),
    ]
    # переопределим window/к через локальные вызовы — проще явно
    def rv(name, cols, **kw):
        edges = window_edges_cols(feats12 if any(c in STATIC for c in cols) else feats,
                                  T_END, pairs, ids, cols, **kw)
        m, q = louvain(len(ids), edges)
        return {"variant": name, "edges": len(edges), "clusters": int(m.max() + 1),
                "clusters_3plus": int(sum((m == c).sum() >= 3 for c in np.unique(m))),
                "q": round(float(q), 3),
                "purity": round(purity(m, ids, agg_by_id), 3),
                "ari": round(ari_all(m, ids, agg_by_id), 3)}

    rows = [
        rv("prod w12", FEAT_COLS, window=12),
        rv("prod w24", FEAT_COLS, window=24),
        rv("prod K15", FEAT_COLS, k=15),
        rv("prod K15 w12", FEAT_COLS, k=15, window=12),
        rv("E1 (8,z) w12", FEAT_COLS + ["log_salary", "log_market_acc"], z=True, window=12),
    ]
    # гибрид весов
    for wd in (0.7, 0.8):
        edges = weighted_edges(feats12, T_END, pairs, ids, wd, window=6)
        m, q = louvain(len(ids), edges)
        rows.append({"variant": f"hyb w_dyn={wd}", "edges": len(edges),
                     "clusters": int(m.max() + 1),
                     "clusters_3plus": int(sum((m == c).sum() >= 3 for c in np.unique(m))),
                     "q": round(float(q), 3),
                     "purity": round(purity(m, ids, agg_by_id), 3),
                     "ari": round(ari_all(m, ids, agg_by_id), 3)})
    g = pd.DataFrame(rows)
    print(g.to_string(index=False))
    g.to_csv("out/features_exp_extra.csv", index=False)


if __name__ == "__main__":
    main()
