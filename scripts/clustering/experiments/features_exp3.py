"""Эксперимент 3: дискриминативные сходства + детерминированный тай-брейк.

Диагноз (см. features_exp): косинусы 6-мес. средневекторов почти все >= 0.999 —
бaskets МО почти параллельны, правило топ-8 вырождается в произвольный
тай-брейк (quicksort), разбег ARI 0.65-0.75.

Варианты (кандидаты <=50 км, топ-8, симметрия, детерминированный
distance-тай-брейк при равных весах):
  S1  косинус raw-средних окна 6 мес. (= production-подобие, но детерминированно)
  S2  косинус z-scored средних окна 6 мес. (= C, детерминированно)
  S3  средняя по 6 признакам Pearson-корреляция 24-мес. месячных рядов
  S4  косинус z-scored средних окна 24 мес.
  S5  средняя по 6 признакам Pearson-корреляция 6-мес. окон (t_end=23)

Метрики: n, q, purity, ARI (все МО перечня). Разбег: по 3 прогона Louvain
(без seed) на фиксированных рёбрах — ARI/purity min-median-max.
Пишет: out/features_exp3.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ..features import FEAT_COLS, load_features
from ..network.graphs import candidate_pairs, louvain
from ..tune import purity
from .features_exp import T_END, zscore

KNN = 8
W = 6


def _sim_matrix(feats: pd.DataFrame, t_end: int, ids: np.ndarray, kind: str) -> np.ndarray:
    if kind in ("s1", "s2", "s4"):
        wlen = W if kind in ("s1", "s2") else 24
        w = feats[(feats.t >= t_end - wlen + 1) & (feats.t <= t_end)]
        prof = w.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0).to_numpy(float)
        if kind == "s2":
            prof = zscore(prof)
        n = np.linalg.norm(prof, axis=1).clip(min=1e-9)
        return (prof / n[:, None]) @ (prof / n[:, None]).T
    # корреляция месячных рядов (z-scored по признакам)
    wlen = W if kind == "s5" else 24
    w = feats[(feats.t >= t_end - wlen + 1) & (feats.t <= t_end)]
    mat = w.pivot_table(index="mo", columns="t", values=FEAT_COLS).reindex(ids)
    X = mat.to_numpy(float).reshape(len(ids), -1, len(FEAT_COLS))  # (n, T, 6)
    X = np.where(np.isnan(X), np.nanmean(X, axis=1, keepdims=True), X)
    Z = X - X.mean(axis=1, keepdims=True)
    sd = Z.std(axis=1, keepdims=True)
    Z = Z / np.clip(sd, 1e-9, None)
    n = Z.shape[0]
    C = np.empty((n, n))
    B = 256
    for s0 in range(0, n, B):
        Zb = Z[s0:s0 + B]  # (B, T, F)
        # среднее по признакам корреляций: (1/((T-1)*F)) * sum_t sum_f Z_i[t,f]*Z_j[t,f]
        C[s0:s0 + B] = np.einsum("bij,kij->bk", Zb, Z) / ((X.shape[1] - 1) * Z.shape[2])
    return C


def build_edges(sim: np.ndarray, pairs: np.ndarray, dist: np.ndarray) -> list[tuple[int, int, float]]:
    # пары bidireктивны — дедупликация до неупорядоченных, иначе топ-8 "съестся" дублями
    lo, hi = np.minimum(pairs[:, 0], pairs[:, 1]), np.maximum(pairs[:, 0], pairs[:, 1])
    keep = {}
    for (u, v), ww, dd in zip(zip(lo, hi), np.arange(len(lo)), dist):
        if (u, v) not in keep or dd < keep[(u, v)]:
            keep[(u, v)] = dd
    u = np.array(list(keep.keys()))  # (N, 2)
    d = np.array(list(keep.values()))
    a, b = u[:, 0], u[:, 1]
    w = sim[a, b].copy()
    df = pd.DataFrame({"i": a, "j": b, "w": w, "d": d})
    df = df[df.w > 0]  # положительное сходство; нулевые/отрицательные — не связь
    both = pd.concat([df, df.assign(i=df.j, j=df.i)], ignore_index=True)
    # детерминированный выбор: по сходству убыв, при равенстве — по расстоянию возр
    both = both.sort_values(["w", "d"], ascending=[False, True], kind="mergesort")
    both = both.groupby("i", sort=False).head(KNN)
    best: dict[tuple[int, int], float] = {}
    for (i, j), ww in zip(zip(both.i.to_numpy(), both.j.to_numpy()), both.w.to_numpy()):
        key = (int(min(i, j)), int(max(i, j)))
        best[key] = max(best.get(key, -1.0), float(ww))
    return [(i, j, ww) for (i, j), ww in best.items() if ww > 0]


def evaluate(name: str, edges: list[tuple[int, int, float]], ids: np.ndarray,
             agg_by_id: dict) -> dict:
    rows = []
    for r in range(3):
        m, q = louvain(len(ids), edges)
        mask = np.array([int(t) in agg_by_id for t in ids])
        from sklearn.metrics import adjusted_rand_score
        y = np.array([agg_by_id[int(t)] for t in ids[mask]])
        rows.append((float(q), int(m.max() + 1),
                     round(purity(m, ids, agg_by_id), 3),
                     round(float(adjusted_rand_score(y, m[mask])), 3)))
    rows = np.array(rows)
    return {"variant": name, "edges": len(edges),
            "q": f"{rows[:,0].min():.3f}-{np.median(rows[:,0]):.3f}-{rows[:,0].max():.3f}",
            "n": int(np.median(rows[:, 1])),
            "purity": f"{rows[:,2].min():.3f}-{np.median(rows[:,2]):.3f}-{rows[:,2].max():.3f}",
            "ari": f"{rows[:,3].min():.3f}-{np.median(rows[:,3]):.3f}-{rows[:,3].max():.3f}"}


def main() -> None:
    feats, _ = load_features()
    ids = np.sort(feats.mo.unique())
    off = pd.read_csv("data/experiments/inputs/derived/official_agglomerations_mapped.csv")
    off = off[off.territory_id.notna()]  # type: ignore[truthy-function]
    agg_by_id = dict(zip(off.territory_id, off["Агломерация"], strict=True))
    pairs, _ = candidate_pairs("data/sources", ids)
    c = pd.read_csv("data/sources/matrix_distance_mo_pairs.csv")
    c = c[c.distance <= 50].set_index(["territory_id_x", "territory_id_y"]).distance
    dmap = c.to_dict()
    dist = np.array([dmap[(int(t1), int(t2))]
                     for t1, t2 in zip(ids[pairs[:, 0]], ids[pairs[:, 1]])], dtype=float)

    rows = []
    for kind, name in [("s1", "S1 cos raw 6m"), ("s2", "S2 cos z 6m"),
                       ("s3", "S3 corr 24m"), ("s4", "S4 cos z 24m"),
                       ("s5", "S5 corr 6m")]:
        sim = _sim_matrix(feats, T_END, ids, kind)
        np.fill_diagonal(sim, -1.0)
        edges = build_edges(sim, pairs, dist)
        rows.append(evaluate(name, edges, ids, agg_by_id))
        print(rows[-1])
    g = pd.DataFrame(rows)
    g.to_csv("out/features_exp3.csv", index=False)
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
