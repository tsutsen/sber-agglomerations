"""Дизайн-эксперименты (чувствительность и независимая валидация):

A) Административный базлайн: насколько рыночные кластеры совпадают с
   административным делением (субъекты) против официального перечня.
   Прямая проверка тезиса «границы не административные»:
   ARI(рынок, перечень) >> ARI(рынок, субъекты).

B) Индекс мобильности Сбера (СЗФО, 288 МО): медианный радиус покупок.
   Читаем по типам кластеров: пригородная периферия рынка (MO официальной
   агломерации + прирост) должен давать больший радиус, чем ядра и сельские.

Читает:  data/*, out/clusters_all.csv, out/cluster_vs_official.csv
Пишет:   out/design_A_baseline.csv, out/design_B_mobility.csv

Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.design.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ..features import load_features


def main() -> None:
    feats, _ = load_features()
    ids = np.sort(feats.mo.unique())
    cl = pd.read_csv("out/clusters_all.csv")
    final = cl[cl.window == cl.window.max()]
    member = final.set_index("mo").cluster.reindex(ids).to_numpy()

    base = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                       usecols=["territory_id", "region_name"])
    region = base.set_index("territory_id").reindex(ids).region_name.to_numpy()
    off = pd.read_csv("data/experiments/inputs/derived/official_agglomerations_mapped.csv")
    off = off[off.territory_id.notna()]
    agg_of = {int(t): a for t, a in zip(off.territory_id, off["Агломерация"], strict=True)}
    official = np.array([agg_of.get(int(t), "вне перечня") for t in ids])

    # A) три взаимных ARI — на одном наборе узлов: МО официального перечня.
    # (по всей выборке 84% МО «вне перечня» дают один общий ярлык и рушат ARI)
    on_list = np.array([int(t) in agg_of for t in ids])
    ari = {
        "рынок vs официальный перечень (на МО перечня)":
            adjusted_rand_score(member[on_list], official[on_list]),
        "рынок vs административное деление (на МО перечня)":
            adjusted_rand_score(member[on_list], region[on_list]),
        "перечень vs административное деление (на МО перечня)":
            adjusted_rand_score(official[on_list], region[on_list]),
    }
    # средняя доля крупнейшего значения референса внутри кластеров (>= 3 МО)
    def top_ref_share(labels: np.ndarray, ref: np.ndarray) -> float:
        vals = []
        for c in np.unique(labels):
            m = labels == c
            if m.sum() < 3:
                continue
            vals.append(max(np.bincount(np.unique(ref[m], return_inverse=True)[1])) / m.sum())
        return float(np.mean(vals))
    ari["доля: один субъект доминирует в кластере (>=3 МО), рынок"] = \
        top_ref_share(member, region)
    ari["доля: одна агломерация доминирует в кластере (>=3 МО), рынок"] = \
        top_ref_share(member, official)

    # B) мобильность СЗФО по типам
    mob = pd.read_csv("data/experiments/inputs/derived/mo_mobility.csv").set_index("territory_id")
    typ = pd.read_csv("out/cluster_vs_official.csv")
    typ_of = dict(zip(typ.cluster, typ.type))
    off_member = set(off.territory_id)
    rows = []
    m2024 = mob.mobility_2024.to_dict()
    covered = [i for i, t in enumerate(ids) if int(t) in m2024]
    if covered:
        X = np.asarray(covered)
        lab = member[X]
        t = ids[X]
        v = np.array([m2024[int(t2)] for t2 in t], float)
        kind = np.array([
            "ядро (МО официальной)" if int(t2) in off_member else "прирост рынка (вне перечня)"
            for t2 in t])
        types = np.array([typ_of.get(int(lab2), "?") for lab2 in lab])
        for typ_name in ["match", "match_partial", "match_expanded", "merge", "new_market"]:
            for k in ["ядро (МО официальной)", "прирост рынка (вне перечня)"]:
                m = (types == typ_name) & (kind == k)
                if m.sum() >= 3:
                    rows.append({"block": "B_mobilnost_szfo",
                                 "segment": f"{typ_name}: {k}",
                                 "n_mo": int(m.sum()),
                                 "median_km_2024": round(float(np.median(v[m])), 2),
                                 "mean_km_2024": round(float(v[m].mean()), 2)})
        rows.append({"block": "B_mobilnost_szfo", "segment": "ВСЕ СЗФО", "n_mo": int(len(v)),
                     "median_km_2024": round(float(np.median(v)), 2),
                     "mean_km_2024": round(float(v.mean()), 2)})

    outA = pd.DataFrame(list(ari.items()), columns=["metric", "value"]).assign(block="A_baseline")
    outB = pd.DataFrame(rows)
    print(outA.to_string(index=False))
    print("\n" + outB.to_string(index=False))
    outA.to_csv("out/design_A_baseline.csv", index=False)
    outB.to_csv("out/design_B_mobility.csv", index=False)


if __name__ == "__main__":
    main()
