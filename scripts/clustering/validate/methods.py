"""Сравнение методов (требование задачи): правило сходства × алгоритм кластеризации.

Варианты:
  сходство:  cos      — косинус сходства профилей (основной вариант);
             pearson  — корреляция Пирсона (профиль центрируется по МО за окно);
  алгоритм:  louvain  — multilevel community detection на графе (основной);
             agglom.  — иерархическая (average linkage, precomputed; число
                        кластеров подобрано равным числу у louvain).

Все варианты — финальное окно (6 мес.), те же кандидаты (50 км) и топ-K = 8.
Метрики: n_clusters, MQ (modularity, только для louvain), ARI против
официального перечня (по МО, входящим в перечень), доля МО в кластерах >= 3.

Читает:  data/sources/* (как основной пайплайн)
Пишет:   out/methods_compare.csv

Запуск из корня:  python -m scripts.clustering.validate.methods  (или: python run_all.py validate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

from ..features import FEAT_COLS, load_features
from ..network.graphs import KNN, candidate_pairs, louvain, window_edges

T_END = 23
WINDOW = 6


def edges_for(feats: pd.DataFrame, ids: np.ndarray, pairs: np.ndarray,
              sim: str) -> list[tuple[int, int, float]]:
    """Рёбра по правилу «кандидаты <= cutoff + топ-K по сходству» для данного сходства."""
    if sim == "cos":
        return window_edges(feats, T_END, pairs, ids, k=KNN, window=WINDOW)
    w = feats[(feats.t >= T_END - WINDOW + 1) & (feats.t <= T_END)]
    prof = w.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0).to_numpy(float)
    prof = prof - prof.mean(axis=1, keepdims=True)  # центрирование -> Пирсон
    norm = np.linalg.norm(prof, axis=1).clip(min=1e-9)
    a, b = pairs[:, 0], pairs[:, 1]
    simv = (prof[a] * prof[b]).sum(1) / (norm[a] * norm[b])
    df = pd.DataFrame({"i": a, "j": b, "w": simv})
    df["i2"], df["j2"] = b, a
    both = pd.concat([df[["i", "j", "w"]], df[["i2", "j2", "w"]]], ignore_index=True)
    both = both.sort_values("w", ascending=False).groupby("i").head(KNN)
    lo, hi = np.minimum(both.i, both.j), np.maximum(both.i, both.j)
    uniq = dict(zip(zip(lo, hi), both.w.to_numpy(), strict=True))
    return [(int(i), int(j), float(v)) for (i, j), v in uniq.items()]


def ari_vs_official(members: np.ndarray, ids: np.ndarray) -> float:
    off = pd.read_csv("data/sources/official_agglomerations.csv").rename(
        columns={"agg_name": "Агломерация"})
    off = off[off.territory_id.notna()]
    ref = {int(t): a for t, a in zip(off.territory_id, off["Агломерация"], strict=True)}
    common = [i for i, t in enumerate(ids) if int(t) in ref]
    y_mar = members[np.asarray(common)]
    y_ref = np.array([ref[int(ids[i])] for i in common])
    return float(adjusted_rand_score(y_mar, y_ref))


def share_3plus(members: np.ndarray) -> float:
    big = np.unique(members)[[(members == c).sum() >= 3 for c in np.unique(members)]]
    return float(np.isin(members, big).mean())


def main() -> None:
    feats, _ = load_features()
    ids = np.sort(feats.mo.unique())
    pairs, _ = candidate_pairs("data/sources", ids)

    rows = []
    for sim in ("cos", "pearson"):
        edges = edges_for(feats, ids, pairs, sim)
        mem_lv, q_lv = louvain(len(ids), edges)
        k = int(len(np.unique(mem_lv)))

        # иерархическая: дистанция = 1 - w на рёбрах, 1 вне графа
        D = np.ones((len(ids), len(ids)))
        a = np.array([e[0] for e in edges])
        b = np.array([e[1] for e in edges])
        wgt = 1 - np.array([e[2] for e in edges], float)
        D[a, b] = wgt
        D[b, a] = wgt
        mem_ag = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                         linkage="average").fit_predict(D)

        for name, mem, q in ((f"{sim} + louvain", mem_lv, float(q_lv)),
                             (f"{sim} + agglomerative", mem_ag, float("nan"))):
            rows.append({"variant": name, "n_clusters": int(len(np.unique(mem))),
                         "MQ_modularity": round(q, 3),
                         "ARI_vs_official": round(ari_vs_official(mem, ids), 3),
                         "share_mo_in_clusters_3plus": round(share_3plus(mem), 3)})
    out = pd.DataFrame(rows)
    out.to_csv("out/methods_compare.csv", index=False)
    print(out.to_string(index=False))
    print("\n-> out/methods_compare.csv")


if __name__ == "__main__":
    main()
