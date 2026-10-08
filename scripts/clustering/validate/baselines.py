"""Null-модели: насколько сигнал профилей вообще нужен.

Четыре графа на одних узлах (финальное окно), одна метрика-набор:
  ours           — правило проекта: <=50 км И топ-8 по косинусу;
  distance_top8  — топ-8 по дорожному расстоянию среди пар <=50 км
                   (честный null той же плотности: география без сходства);
  distance_only  — все пары <=50 км (география без сходства профилей);
  attribute_only — топ-8 по косинусу БЕЗ ограничения расстояния;
  random         — столько же случайных рёбер, сколько в ours (seed=42).

Метрики: число кластеров, модулярность, purity (как в tune.py),
ARI против официального перечня (на всех МО перечня).

Если ARI/purity у ours заметно выше остальных — правило «география +
сходство» несёт информацию; если нет — сходство профилей лишнее.

Читает:  data/sources/*, out/*
Пишет:   out/baselines.csv

Запуск из корня:  python -m scripts.clustering.validate.baselines  (или: python run_all.py validate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ..features import FEAT_COLS, load_features
from ..network.graphs import candidate_pairs, louvain, window_edges


def attribute_only_edges(feats: pd.DataFrame, t_end: int, ids: np.ndarray, k: int = 8):
    """Топ-k по косинусу на ВСЕХ парах (без distance-cutoff)."""
    w = feats[(feats.t >= t_end - 5) & (feats.t <= t_end)]
    prof = w.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0).to_numpy()
    norm = np.linalg.norm(prof, axis=1).clip(min=1e-9)
    cos = (prof @ prof.T) / np.outer(norm, norm)
    np.fill_diagonal(cos, -np.inf)
    idx = np.argsort(-cos, axis=1)[:, :k]
    rows = [(i, j, float(cos[i, j])) for i in range(len(ids)) for j in idx[i]]
    df = pd.DataFrame(rows, columns=["a", "b", "w"])
    lo, hi = np.minimum(df.a, df.b), np.maximum(df.a, df.b)
    uniq = dict(zip(zip(lo, hi), df.w))
    return [(int(i), int(j), float(w)) for (i, j), w in uniq.items()]


def distance_top8_edges(pairs: np.ndarray, ids: np.ndarray, k: int = 8) -> list[tuple[int, int, float]]:
    """Топ-k ближайших (по дорожному расстоянию) соседей на <=50 км, симметрично."""
    # та же дорожная матрица, что и финальный пайплайн (OSM; см. data/sources/README.md)
    c = pd.read_csv("data/sources/matrix_distance_mo_pairs.csv")
    c = c[c.distance <= 50].set_index(["territory_id_x", "territory_id_y"]).distance
    dmap = c.to_dict()
    df = pd.DataFrame({"i": pairs[:, 0], "j": pairs[:, 1]})
    t1, t2 = ids[pairs[:, 0]].astype(int), ids[pairs[:, 1]].astype(int)
    df["d"] = [dmap.get((a, b), dmap.get((b, a))) for a, b in zip(t1, t2)]
    df = df[df.d.notna()]
    both = pd.concat([df, df.assign(i=df.j, j=df.i)], ignore_index=True)
    both = both.sort_values("d", ascending=True, kind="mergesort").groupby("i", sort=False).head(k)
    lo, hi = np.minimum(both.i.to_numpy(), both.j.to_numpy()), np.maximum(both.i.to_numpy(), both.j.to_numpy())
    uniq = dict(zip(zip(lo, hi), both.d.to_numpy(), strict=False))
    return [(int(i), int(j), 1.0 / max(float(w), 1e-6)) for (i, j), w in uniq.items()]


def main() -> None:
    feats, _ = load_features()
    ids = np.sort(feats.mo.unique())
    t_end = 23
    off = pd.read_csv("data/sources/official_agglomerations.csv").rename(
        columns={"agg_name": "Агломерация"})
    off = off[off.territory_id.notna()]
    agg_by_id = dict(zip(off.territory_id, off["Агломерация"], strict=True))
    on_list = [int(t) in agg_by_id for t in ids]
    official = np.array([agg_by_id.get(int(t), "вне перечня") for t in ids])
    from ..tune import purity  # та же формула, что в подборе гиперпараметров

    pairs, _ = candidate_pairs("data/sources", ids)
    variants = {
        "ours": window_edges(feats, t_end, pairs, ids, k=8),
        "distance_top8": distance_top8_edges(pairs, ids, k=8),
        "distance_only": [(int(a), int(b), 1.0) for a, b in pairs],
        "attribute_only": attribute_only_edges(feats, t_end, ids),
        "random": None,  # ниже
    }
    n_ours = len(variants["ours"])
    rng = np.random.default_rng(42)
    a = rng.integers(0, len(ids), n_ours)
    b = rng.integers(0, len(ids), n_ours)
    m = a != b
    pairs_r = np.column_stack([a[m], b[m]])
    lo, hi = np.minimum(pairs_r[:, 0], pairs_r[:, 1]), np.maximum(pairs_r[:, 0], pairs_r[:, 1])
    seen = set(zip(lo, hi))
    variants["random"] = [(int(i), int(j), 1.0) for (i, j) in seen]

    rows = []
    members = {}
    for name, edges in variants.items():
        member, q = louvain(len(ids), edges)
        members[name] = member
        n3 = sum(1 for c in np.unique(member) if (member == c).sum() >= 3)
        rows.append({
            "variant": name,
            "edges": len(edges),
            "clusters": len(np.unique(member)),
            "clusters_3plus": n3,
            "q": round(float(q), 3),
            "purity": round(purity(member, ids, agg_by_id), 3),
            "ARI_vs_official": round(float(adjusted_rand_score(member[on_list], official[on_list])), 3),
        })
    out = pd.DataFrame(rows).set_index("variant")
    print(out.to_string())
    out.to_csv("out/baselines.csv")

    # ARI между вариантами (наши четыре метода из methods.py + null-модели)
    mp = pd.DataFrame({n: m for n, m in members.items()})
    import itertools
    pw = {}
    for x, y in itertools.combinations(mp.columns, 2):
        pw[f"{x} vs {y}"] = round(float(adjusted_rand_score(mp[x], mp[y])), 3)
    pd.Series(pw, name="ARI").to_csv("out/baselines_pairwise_ARI.csv")
    print("\nPairwise ARI:")
    print(pd.Series(pw).to_string())


if __name__ == "__main__":
    main()
