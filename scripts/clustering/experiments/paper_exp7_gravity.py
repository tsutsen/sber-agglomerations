"""EXP 7: гравитационное перерейтинг рёбер (motif: gravity null, Gagolewski 2022).

Production-правило выбирает top-8 по косинусу профиля — но косинус плато
(>=0.999 у 99.8% пар), т.е. различает только на 3-4-м знаке; фактический
сигнал про «насколько похожи» несёт лишь хвост. Гипотеза: вычесть
гравитационную составляющую g(d) (средний косинус пар на расстоянии d)
и ранжировать пары по «избытку сходства» над гравитационным ожиданием:
  score = cos - g(d)   или   score = cos / g(d).
Правило: top-8 на МО по score (вместо top-8 по cos), дальше тот же
Louvain. Базовая линия — top-8 по cos (тот же конвейер).

Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.paper_exp7_gravity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features_exp import ari_all
from .papers_common import coords, haversine, load_ctx
from ..features import FEAT_COLS
from ..network.graphs import D_CUTOFF, KNN, WINDOW, louvain
from ..tune import purity

T_END = 23


def prod_edges_scored(feats: pd.DataFrame, t_end: int, pairs: np.ndarray,
                      ids: np.ndarray, score: np.ndarray,
                      k: int = KNN) -> list[tuple[int, int, float]]:
    """Точная копия production network.graphs.window_edges, но топ-k выбирается
    по score (может быть cos, cos-g, cos/g); вес ребра — cos (как в prod)."""
    w = feats[(feats.t >= t_end - WINDOW + 1) & (feats.t <= t_end)]
    prof = w.groupby("mo")[FEAT_COLS].mean()
    prof = prof.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    norm = np.linalg.norm(prof, axis=1).clip(min=1e-9)
    a, b = pairs[:, 0], pairs[:, 1]
    cos = (prof[a] * prof[b]).sum(1) / (norm[a] * norm[b])
    df = pd.DataFrame({"i": a, "j": b, "w": cos, "s": score})
    df2 = df
    df2["i2"], df2["j2"] = b, a
    both = pd.concat([df[["i", "j", "w", "s"]],
                      df2[["i2", "j2", "w", "s"]]], ignore_index=True)
    both = both.sort_values("s", ascending=False).groupby("i").head(k)
    lo, hi = np.minimum(both.i, both.j), np.maximum(both.i, both.j)
    uniq = dict(zip(zip(lo, hi), both.w.to_numpy(), strict=True))
    return [(int(i), int(j), float(w)) for (i, j), w in uniq.items()]


def main() -> None:
    feats, ids, pairs, agg_by_id = load_ctx()
    w = feats[(feats.t >= T_END - WINDOW + 1) & (feats.t <= T_END)]
    prof = w.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0)
    prof = prof.to_numpy(dtype=float)
    norm = np.linalg.norm(prof, axis=1).clip(min=1e-9)
    a, b = pairs[:, 0], pairs[:, 1]
    cos = (prof[a] * prof[b]).sum(1) / (norm[a] * norm[b])

    lat, lon = coords(ids)
    d = haversine(lat, lon)[a, b]
    bin = pd.cut(d, bins=np.arange(0, D_CUTOFF + 5, 5))
    g = pd.Series(cos).groupby(bin).transform("mean").to_numpy()
    g = np.where(np.isfinite(g), g, cos.mean())

    rows = []
    for name, score in [("prod-правило, top8 по cos (база)", cos),
                        ("prod-правило, score = cos - g(d)", cos - g),
                        ("prod-правило, score = cos / g(d)", cos / g)]:
        edges = prod_edges_scored(feats, T_END, pairs, ids, score)
        members, q = louvain(len(ids), edges)
        rows.append({"variant": name, "edges": len(edges),
                     "clusters": int(members.max() + 1),
                     "clusters_3plus": int(sum((members == c).sum() >= 3
                                               for c in np.unique(members))),
                     "q": round(float(q), 3),
                     "purity": round(purity(members, ids, agg_by_id), 3),
                     "ari": round(ari_all(members, ids, agg_by_id), 3)})
    out = pd.DataFrame(rows)
    out.to_csv("out/paper_exp7_gravity.csv", index=False)
    print(out.to_string(index=False))
    print("\ng(d) по бидам (км):")
    print(pd.Series(cos).groupby(bin).mean().round(4).to_string())


if __name__ == "__main__":
    main()
