"""EXP 6: расширенный k-means в дуальном пространстве (Gagolewski et al. 2022).

Статья: кластеризовать одновременно в пространстве признаков и в
пространстве сети, комбинируя метрики. Практическая реализация:
  d_comb(i,j) = (1-lam) * d_feat(i,j) + lam * d_net(i,j),
  d_feat — евклидово расстояние в z-пространстве 14 осей,
  d_net  — 1 - косинус-14 (сеть похожести профилей);
  обе метрики нормированы на свой максимум; k-means (K=15) в 50-мерном
  MDS-вложении d_comb, тёплый старт из типологии, 50 итераций.

Сравниваем с базовой типологией (mutual-kNN + Louvain): spatial_coh +
null/p, ARI, силуэт в комбинированном пространстве, PP.

Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.paper_exp6_kmeans.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from .features_exp import ari_all
from .papers_common import (coords, load_ctx, neighbors, spatial_coh,
                                   typ14)

LAMS = [0.25, 0.5, 0.75]
K = 15
N_NULL = 20


def main() -> None:
    feats, ids, pairs, agg_by_id = load_ctx()
    lat, lon = coords(ids)
    X, cols, mt, q = typ14(feats, ids, pairs)
    Xz = StandardScaler().fit_transform(X)
    n = len(ids)

    norm = np.linalg.norm(Xz, axis=1).clip(min=1e-9)
    S = np.clip((Xz / norm[:, None]) @ (Xz / norm[:, None]).T, 0.0, None)
    np.fill_diagonal(S, 0.0)

    d_feat = np.sqrt(((Xz[:, None, :] - Xz[None, :, :]) ** 2).sum(-1))
    d_feat = (d_feat + d_feat.T) / 2
    d_net = 1.0 - S
    d_net = (d_net + d_net.T) / 2
    np.fill_diagonal(d_feat, 0.0)
    np.fill_diagonal(d_net, 0.0)

    nb = neighbors(pairs)
    base_coh, base_null, base_p = spatial_coh(mt, nb)
    ari_base = ari_all(mt, ids, agg_by_id)

    rows = [{"variant": "typology (kNN+Louvain, base)", "lam": np.nan,
             "K": K, "spatial_coh": round(base_coh, 3),
             "null_max": round(base_null, 3), "p": round(base_p, 3),
             "ARI": round(ari_base, 3)}]
    for lam in LAMS:
        d_comb = ((1 - lam) * d_feat / d_feat.max()
                  + lam * d_net / d_net.max())
        d_comb = (d_comb + d_comb.T) / 2
        np.fill_diagonal(d_comb, 0.0)
        emb = MDS(n_components=50, dissimilarity="precomputed",
                  normalized_stress=False, random_state=7).fit_transform(d_comb)
        # тёплый старт: центры = средние по меткам базовой типологии
        init = np.array([emb[mt == k].mean(axis=0) for k in range(K)])
        km = KMeans(n_clusters=K, n_init=1, init=init, max_iter=50, random_state=7)
        lab = km.fit_predict(emb)
        coh, null_max, p = spatial_coh(lab, nb)
        sil = _sil(d_comb, lab)
        ari_x = ari_all(lab, ids, agg_by_id)
        rows.append({"variant": f"extended k-means (lam={lam})", "lam": lam,
                     "K": K, "spatial_coh": round(coh, 3),
                     "null_max": round(null_max, 3), "p": round(p, 3),
                     "ARI": round(ari_x, 3), "sil_comb": round(sil, 3)})

    out = pd.DataFrame(rows)
    out.to_csv("out/paper_exp6_kmeans.csv", index=False)
    print(out.to_string(index=False))


def _sil(d: np.ndarray, lab: np.ndarray) -> float:
    D = d
    n = len(lab)
    K = lab.max() + 1
    a = np.zeros((n, K))
    for k in range(K):
        mk = lab == k
        a[mk, k] = D[np.ix_(mk, mk)].sum(axis=1) / max(int(mk.sum()) - 1, 1)
    b = np.zeros((n, K - 1))
    ks = list(range(K))
    for i in range(n):
        ki = lab[i]
        others = [k for k in ks if k != ki]
        for j, k in enumerate(others):
            mk = lab == k
            b[i, j] = D[i, mk].sum() / max(int(mk.sum()), 1)
    b.sort(axis=1)
    b = b[:, 0]
    s = (b - a[:, lab]) / np.maximum(a[:, lab] + b, 1e-9)
    return float(np.nanmean(s))


if __name__ == "__main__":
    main()
