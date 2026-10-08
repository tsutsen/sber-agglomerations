"""Кластеризация по fill_real (реальная покупательная способность).

Сравнение с типологией-14: насколько 1-2 мерные решения по fill_real
пространственно когерентны (p-value) и сколько дисперсии fill_real
объясняют (eta2). Плюс контроль: по зарплате и по цене отдельно.

Вывод: out/basket_cluster.csv, out/basket_cluster_results.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def main() -> None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from ..network.graphs import candidate_pairs

    mo = pd.read_csv("out/basket_index_mo.csv")
    d = mo.dropna(subset=["coef24", "fill_real"]).copy()
    print("MO с данными:", len(d))

    pairs, _ = candidate_pairs("data/sources", np.arange(2144))  # соседи <=50 км
    nb: dict[int, set[int]] = {}
    for a, b in pairs:
        a, b = int(a), int(b)
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    ids = d.territory_id.to_numpy()
    pos = {int(t): i for i, t in enumerate(ids)}

    def coherence(m: np.ndarray) -> float:
        tot = hit = 0
        for i in range(len(m)):
            ns = [pos[j] for j in nb.get(i, ()) if j in pos]
            if not ns:
                continue
            tot += 1
            if any(m[j] == m[i] for j in ns):
                hit += 1
        return hit / tot

    rng = np.random.default_rng(42)

    def evaluate(col: str, k: int) -> dict:
        x = d[col].to_numpy(dtype=float)[:, None]
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(x)
        m = km.labels_
        sil = silhouette_score(x, m)
        coh = coherence(m)
        null = max(coherence(rng.permutation(m)) for _ in range(500))
        eta2 = ((np.bincount(m, weights=np.ones(len(m))) *
                 (d.groupby(km.labels_)[col].mean().to_numpy() - d[col].mean()) ** 2).sum() /
                ((len(m) - 1) * d[col].var(ddof=1)))
        size = np.bincount(m)
        return {"k": k, "sil": round(float(sil), 3), "coh": round(float(coh), 3),
                "coh_null_max": round(float(null), 3),
                "eta2": round(float(eta2), 3), "sizes": (int(size.max()), int(size.min()))}

    res: dict[str, list] = {}
    for col in ("fill_real", "salary", "fill_price"):
        res[col] = [evaluate(col, k) for k in (5, 10, 15, 17)]
        print(col, res[col])

    # финальное решение: fill_real, k=10 (компромисс; см. results)
    k = 10
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(d[["fill_real"]])
    d = d.copy()
    d["basket_cluster"] = km.labels_
    out = d[["territory_id", "region_name", "municipal_district_name", "fill_price",
             "fill_real", "basket_cluster"]]
    out.to_csv("out/basket_cluster.csv", index=False)

    # профиль кластеров
    prof = d.groupby("basket_cluster").agg(
        n=("territory_id", "size"), fill_real=("fill_real", "mean"),
        fill_price=("fill_price", "mean"), salary=("salary", "mean")).round(1)
    json.dump({"results": res, "final_k": k,
               "final": res["fill_real"][3],  # k=10 строка
               "profile": prof.reset_index().to_dict("records")},
              open("out/basket_cluster_results.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nфинальный профиль (fill_real k=%d):" % k)
    print(prof.sort_values("fill_real").to_string())


if __name__ == "__main__":
    main()
