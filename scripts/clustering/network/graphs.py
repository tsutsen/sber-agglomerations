"""Динамический граф: кандидаты (дорога <= cutoff) + топ-k по сходству профиля.

Правило ребра: МО A соединяется с МО B, если
  1) расстояние по дорогам A–B <= D_CUTOFF км (матрица расстояний OSM);
  2) B входит в топ-K самых похожих на A соседей среди кандидатов.
Сходство = косинус средневектора профиля (5 долей + log уровня) за окно W месяцев.

ПОВЕДЕНИЕ (важно для честного описания в отчёте): candidate_pairs содержит
каждую неупорядоченную пару ОДНУЖДЫ (направление из таблицы связей). Топ-K
выбирается по «своей» строке пары (сторона i); вторая сторона concat
молча отбрасывается (NaN-ключ в groupby). Итог: ~53% МО активно выбирают
свои топ-K, остальные получают рёбра от соседей-выбирающих + дедупликация.
Симметричная версия (оба направления) при тех же признаках даёт ARI ниже —
см. report §5.5 и out/features_exp3.csv.
"""
from __future__ import annotations

import igraph as ig
import numpy as np
import pandas as pd

from ..config import CFG
from ..features import FEAT_COLS

# значения из config.yaml (единственный источник)
D_CUTOFF = CFG["network"]["d_cutoff_km"]  # км — компактный городской район (масштаб агломерации)
KNN = CFG["network"]["knn"]
WINDOW = CFG["network"]["window_months"]    # мес. в скользящем окне
STEP = CFG["network"]["step_months"]        # шаг окон (квартально)


def candidate_pairs(data_dir: str, ids: np.ndarray, cutoff: float = D_CUTOFF) -> tuple[np.ndarray, dict]:
    """Матрица (n,2) индексов пар <= cutoff; id2i — mapping territory_id -> позиция."""
    id2i = {int(t): i for i, t in enumerate(ids)}
    c = pd.read_csv(f"{data_dir}/matrix_distance_mo_pairs.csv")
    c = c[c.distance <= cutoff]
    mx = c.territory_id_x.map(id2i)
    my = c.territory_id_y.map(id2i)
    m = mx.notna() & my.notna()
    pairs = np.column_stack([mx.loc[m].to_numpy(), my.loc[m].to_numpy()]).astype(np.int64)
    return pairs, id2i


def window_edges(feats: pd.DataFrame, t_end: int, pairs: np.ndarray, ids: np.ndarray,
                 k: int = KNN, window: int = WINDOW) -> list[tuple[int, int, float]]:
    """Косинус-сходство на средневекторах за окно (t_end-window+1 .. t_end) + топ-k."""
    w = feats[(feats.t >= t_end - window + 1) & (feats.t <= t_end)]
    prof = w.groupby("mo")[FEAT_COLS].mean()
    prof = prof.reindex(ids).fillna(0.0).to_numpy(dtype=float)  # порядок строк = ids
    norm = np.linalg.norm(prof, axis=1).clip(min=1e-9)
    a, b = pairs[:, 0], pairs[:, 1]
    cos = (prof[a] * prof[b]).sum(1) / (norm[a] * norm[b])
    # топ-k по сходству для каждого узла (в обе стороны), дедупликация
    df = pd.DataFrame({"i": a, "j": b, "w": cos})
    df2 = df
    df2["i2"], df2["j2"] = b, a
    both = pd.concat([df[["i", "j", "w"]], df2[["i2", "j2", "w"]]], ignore_index=True)
    both = both.sort_values("w", ascending=False).groupby("i").head(k)
    bi, bj = both.i.to_numpy(), both.j.to_numpy()
    lo, hi = np.minimum(bi, bj), np.maximum(bi, bj)
    uniq = dict(zip(zip(lo, hi), both.w.to_numpy(), strict=True))
    return [(int(i), int(j), float(w)) for (i, j), w in uniq.items()]


def louvain(n: int, edges: list[tuple[int, int, float]]) -> tuple[np.ndarray, float]:
    g = ig.Graph(n, [(e[0], e[1]) for e in edges])
    g.es["weight"] = [e[2] for e in edges]
    cl = g.community_multilevel(weights="weight")
    return np.asarray(cl.membership), float(cl.q)
