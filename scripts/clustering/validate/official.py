"""Сравнение найденных кластеров с официальным перечнем агломераций.

Типы кластеров (главная идея проекта), полное покрытие случаев:
  1. match          — состав почти совпадает с одной официальной агломерацией;
  2. match_partial  — кластер < одной официальной: рынок покрывает лишь часть её МО;
  3. match_expanded — вся официальная в кластере, но рынок прицепил лишние МО
     (>= 30% состава не в официальном перечне);
  3. merge          — кластер объединяет МО из нескольких официальных;
  4. new_market     — кластер без официального совпадения;
  5. small          — 1-2 МО (слабая агломерация, не «шум» обязательно).

Отдельно (официальная сторона, official_coverage.csv): scattered — официальная
агломерация, МО которой рынок разнёс по нескольким кластерам.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def build_report(clusters: pd.DataFrame, ids: np.ndarray, off: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """clusters: колонки [mo, window, cluster]; рассматривается финальное окно."""
    final = clusters[clusters.window == clusters.window.max()]
    members = final.set_index("mo").cluster.to_numpy()
    agg_by_id = dict(zip(off.territory_id, off["Агломерация"], strict=True))
    # Покрытие считаем по официальным МО, которые ЕСТЬ в наших данных:
    # например, у Мурманской из 3 МО отсутствует ЗАТО Североморск, и кластер
    # из двух оставшихся должен считаться полным совпадением, а не «частью».
    id_set = set(ids.tolist())
    agg_totals = off[off.territory_id.isin(id_set)].groupby("Агломерация").territory_id.count().to_dict()
    rows = []
    for cid in np.unique(members):
        mem = members == cid
        tids = ids[mem]
        counts: dict[str, int] = {}
        for t in tids:
            a = agg_by_id.get(int(t))
            if a:
                counts[a] = counts.get(a, 0) + 1
        n_mo, n_off = int(mem.sum()), int(sum(counts.values()))
        if len(counts) > 1:
            typ = "merge"
        elif len(counts) == 1:
            agg_name = max(counts, key=counts.__getitem__)
            covered = counts[agg_name] / max(agg_totals.get(agg_name, 1), 1)
            extra = n_mo - n_off
            if covered < 0.7:
                typ = "match_partial"    # рынок покрывает лишь часть официальной
            elif extra / n_mo >= 0.3:
                typ = "match_expanded"   # вся официальная + прирост
            else:
                typ = "match"            # почти совпадает
        else:
            typ = "new_market"  # включает одиночные МО вне официального перечня
        rows.append({
            "cluster": int(cid), "n_mo": n_mo,
            "n_official_mo": n_off,
            "n_official_agg": int(len(counts)),
            "dominant_agg": max(counts, key=counts.__getitem__) if counts else "",
            "dominant_share": round(max(counts.values()) / n_mo, 3) if counts else 0.0,
            "type": typ,
        })
    rep = pd.DataFrame(rows)

    # покрытие с другой стороны: для каждой официальной агломерации — куда разнесены её МО
    off_rows = []
    for agg, g in off.groupby("Агломерация"):
        tids = set(g.territory_id)
        id_set = set(ids)
        lab = pd.Series({int(t): members[np.where(ids == t)[0][0]] for t in tids if t in id_set})
        if lab.empty:
            off_rows.append({"agglomeration": agg, "n_official_mo": len(tids), "n_in_market_clusters": 0,
                             "dominant_cluster": -1, "share_in_top_cluster": 0.0, "scattered": False})
            continue
        vc = lab.value_counts()
        dom_lab, n_dom = vc.index[0], vc.max()
        off_rows.append({"agglomeration": agg, "n_official_mo": len(tids),
                         "n_in_market_clusters": int(lab.size),
                         "dominant_cluster": int(dom_lab),  # type: ignore[arg-type]
                         "share_in_top_cluster": round(n_dom / max(lab.size, 1), 3),
                         "scattered": bool(lab.nunique() > 1 and n_dom / max(lab.size, 1) < 0.8)})
    off_rep = pd.DataFrame(off_rows)

    os.makedirs(out_dir, exist_ok=True)  # type: ignore[call-overload]
    rep.to_csv(f"{out_dir}/cluster_vs_official.csv", index=False)
    off_rep.to_csv(f"{out_dir}/official_coverage.csv", index=False)
    return rep


def main() -> None:
    clusters = pd.read_csv("out/clusters_all.csv")
    feats_ids = clusters.mo.unique()
    ids = np.sort(feats_ids)
    off = pd.read_csv("data/sources/official_agglomerations.csv").rename(
        columns={"agg_name": "Агломерация"})
    off = off[off.territory_id.notna()]  # type: ignore[truthy-function]
    rep = build_report(clusters, ids, off, "out")  # type: ignore[arg-type]
    print(rep[rep.type != "small"].groupby("type").n_mo.agg(["count", "sum"]).to_string())
    print("\nМерджи (рынок объединяет официально разные агломерации):")
    print(rep[rep.type == "merge"].sort_values("n_mo", ascending=False).to_string(index=False))  # type: ignore[call-overload]
    print("\nОфициальные агломерации, разнесённые рынком:")
    off_rep = pd.read_csv("out/official_coverage.csv")
    print(off_rep[off_rep.scattered].sort_values("n_official_mo", ascending=False).to_string(index=False))  # type: ignore[call-overload]


if __name__ == "__main__":
    main()
