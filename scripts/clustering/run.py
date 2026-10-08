"""Пайплайн: фичи -> окна 6 мес. (шаг 3) -> cos-граф -> Louvain -> Jaccard -> сравнение с официальным перечнем."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .features import load_features
from .network.graphs import KNN, STEP, WINDOW, candidate_pairs, louvain, window_edges

OUT = "out"


def jaccard_matrix(members: list[np.ndarray], n_windows: int, labels: list[str]) -> pd.DataFrame:
    """Jaccard пересечения кластеров соседних окон."""
    rows = []
    for t in range(1, n_windows):
        prev = np.asarray(members[t - 1])
        cur = np.asarray(members[t])
        pu, pi = np.unique(prev, return_inverse=True)
        cu, ci = np.unique(cur, return_inverse=True)
        cooc = np.zeros((pu.size, cu.size))
        np.add.at(cooc, (pi, ci), 1)
        np_p = np.bincount(pi)
        np_c = np.bincount(ci)
        j = cooc / np.maximum(np_p[:, None] + np_c[None, :] - cooc, 1)
        for jcol, cid in enumerate(cu):
            if np_c[jcol] < 3:
                continue
            b = int(np.argmax(j[:, jcol]))
            rows.append({"t": labels[t], "cluster": int(cid), "n": int(np_c[jcol]),
                         "prev_cluster": int(pu[b]), "jaccard": round(float(j[b, jcol]), 3)})
    return pd.DataFrame(rows)


def compare_official(members: np.ndarray, names: dict, off: pd.DataFrame, label: str) -> pd.DataFrame:
    """Покрытие каждого найденного кластера официальными агломерациями (финальное окно)."""
    by_id = dict(zip(off.territory_id, off["Агломерация"], strict=True))
    rows = []
    for cid in np.unique(members):
        mem = members == cid
        if mem.sum() < 3:
            continue
        tids = np.array(list(names.keys()))[mem]
        labels = pd.Series([by_id.get(t) for t in tids]).dropna()
        top = labels.value_counts().head(5)
        rows.append({"window": label, "cluster": int(cid), "n": int(mem.sum()),
                     "n_official_mo": int(labels.size),
                     "top_agglom": "; ".join(f"{k} ({v})" for k, v in top.items()),
                     "coverage": round(float(labels.size) / mem.sum(), 3)})
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    feats, months = load_features()
    n = feats.mo.nunique()
    ids = np.sort(feats.mo.unique())  # порядок строк профилей = порядок группировки
    names = dict(zip(ids,
                     pd.read_csv("data/sources/derived/mo_consumption_population.csv")
                     .set_index("territory_id").loc[ids, "municipal_district_name"].values,
                     strict=True))
    print(f"МО в работе: {n}")

    pairs, _ = candidate_pairs("data/sources", ids)
    print(f"кандидатных пар (дорога <= cutoff): {len(pairs)}")

    t_ends = list(range(WINDOW - 1, 24, STEP))
    labels = [f"{months[t]}" for t in t_ends]
    all_clusters, members_list = [], []
    for t, lab in zip(t_ends, labels, strict=True):
        edges = window_edges(feats, t, pairs, ids, k=KNN, window=WINDOW)
        member, q = louvain(n, edges)
        members_list.append(member)
        all_clusters.append(pd.DataFrame({"mo": ids, "window": lab, "cluster": member}))
        pd.DataFrame(edges, columns=["i", "j", "w"]).to_csv(f"{OUT}/edges_{lab}.csv", index=False)
        print(f"{lab}: ребер {len(edges)}, кластеров {len(np.unique(member))}, q={q:.3f}")

    cl = pd.concat(all_clusters, ignore_index=True)
    cl.to_csv(f"{OUT}/clusters_all.csv", index=False)
    j = jaccard_matrix(members_list, len(t_ends), labels)
    j.to_csv(f"{OUT}/jaccard.csv", index=False)
    print("\nСредний Jaccard кластеров между окнами:",
          round(float(j.jaccard.mean()), 3) if len(j) else "n/a")

    off = pd.read_csv("data/sources/official_agglomerations.csv")
    off = off.rename(columns={"agg_name": "Агломерация"})
    off = off[off.territory_id.notna() & off.territory_id.isin(set(ids))]
    cmp_ = compare_official(members_list[-1], {int(k): v for k, v in names.items()}, off, labels[-1])
    cmp_.to_csv(f"{OUT}/compare_final.csv", index=False)
    print(f"\nКластеров >=3 МО в финальном окне: {len(cmp_)}; "
          f"с перекрытием официального перечня: {int((cmp_.n_official_mo > 0).sum())}")
    print(cmp_.sort_values("n", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
