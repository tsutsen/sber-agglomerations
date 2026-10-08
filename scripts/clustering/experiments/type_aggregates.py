"""Пересчёт характеристик 17 типов по фактическому по-МО назначению (nearest-centroid).

В out/basket_index_mo.csv тип МО назначен по ближайшему центроиду типов G3,
а out/basket_economy_clusters.csv исторически собирался по членству G3
(types_profile_g3.csv) — отсюда расхождение n_mo и «ядер» с картой.
Скрипт пересобирает агрегат из того же по-МО назначения, что используется
в basket_final.csv, economic_clusters.geojson и лендинге.

Вывод: out/basket_economy_clusters.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MON = {n: i + 1 for i, n in enumerate(
    ["январь", "февраль", "март", "апрель", "май", "июнь",
     "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"])}
STATIC = ["log_salary", "work_share", "mig_per_1000", "log_market_acc",
          "cons_growth", "season_amp"]
# порядок = FEAT_COLS (доли, затем log_total) + STATIC + корзина
AX = ["share_food", "share_health", "share_cafe", "share_transport",
      "share_marketplace", "log_total"] + STATIC + \
     ["log_basket", "growth_basket"]


def main() -> None:
    mo = pd.read_csv("out/basket_index_mo.csv").set_index("territory_id")

    # --- z-профиль по МО (та же процедура, что в basket_index.py)
    import geopandas as gpd
    g = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                    usecols=["territory_id", "region_name", "municipal_district_name"]) \
        .drop_duplicates("territory_id")
    lng = pd.read_csv("data/sources/derived/mo_consumption_monthly_long.csv", usecols=["territory_id"])
    g = g[g.territory_id.isin(set(lng.territory_id.unique()))]
    g = g.set_index("territory_id")

    from ..features import FEAT_COLS, load_features
    f2, _ = load_features()
    dyn = f2.groupby("mo")[FEAT_COLS].mean()
    static = pd.read_csv("data/experiments/inputs/derived/mo_static_features.csv").set_index("territory_id")

    bb = pd.read_csv("data/sources/basket_size_per_mo.csv")
    bb["t"] = (bb.year - 2023) * 12 + bb.month.map(MON) - 1
    bwide = bb.pivot_table(index="region_name", columns="t", values="value")
    bmo = pd.concat([g.region_name.map(bwide[t]).rename(int(t)) for t in bwide.columns],
                    axis=1).reindex(g.index)
    bmo = bmo.fillna(bmo.median(axis=1))
    sb = static.copy()
    sb["log_basket"] = np.log1p(bmo.mean(axis=0))
    sb["growth_basket"] = (bmo.iloc[13:25].mean(axis=0) /
                           bmo.iloc[1:13].mean(axis=0).clip(lower=1e-9) - 1)
    stat8 = sb[STATIC + ["log_basket", "growth_basket"]]

    ids = mo.index.to_numpy()
    assert set(ids) <= set(g.index) and set(ids) <= set(dyn.index)
    assert list(FEAT_COLS) + STATIC + ["log_basket", "growth_basket"] == AX
    Xd = dyn.reindex(ids).fillna(0.0).to_numpy(float)
    Xs = stat8.reindex(ids).fillna(0.0).to_numpy(float)
    X = np.hstack([Xd, Xs])
    X = (X - X.mean(0)) / X.std(0, ddof=0)
    z = pd.DataFrame(X, index=ids, columns=AX)

    # --- ядра: топ-3 МО по экономической активности
    act = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                      usecols=["territory_id", "cons_total_2024", "pop_2024"])
    act["act"] = act.cons_total_2024.fillna(0) * act.pop_2024.fillna(0)
    act = act.set_index("territory_id")
    from .features_exp import mo_names
    names = mo_names()

    # --- агрегация по фактическому назначению типов
    m = mo.join(z)
    rows = []
    for t, grp in m.groupby("type"):
        zt = grp[AX].mean()
        top3 = " > ".join(zt.reindex(zt.abs().sort_values(ascending=False).index).index[:3])
        core = act.reindex(grp.index).act.sort_values(ascending=False).head(3)
        rows.append({
            "economy_cluster": t,
            "n_mo": len(grp),
            "n_regions": grp.region_name.nunique(),
            "cores": " — ".join(names.get(int(i), str(i)) for i in core.index),
            **{c: round(zt[c], 2) for c in AX},
            "price_index_2024": round(float(grp.coef24.mean()), 3),
            "basket_fill": round(float(grp.fill_price.mean()), 1),
            "salary": round(float(grp.salary.mean()), 0),
            "salary_index": round(float(grp.salary_index.mean()), 1),
            "basket_fill_real": round(float(grp.fill_real.mean()), 1),
            "hours_for_1000": round(float(grp.hours_for_1000.mean()), 2),
            "top3": top3,
        })
    ec = pd.DataFrame(rows).sort_values("economy_cluster").reset_index(drop=True)
    old = pd.read_csv("out/basket_economy_clusters.csv")
    ec = ec.merge(old[["economy_cluster", "type_name"]], on="economy_cluster", how="left")
    assert ec.type_name.notna().all()
    ec = ec[["economy_cluster", "type_name", "n_mo", "n_regions", "cores",
             *AX, "price_index_2024", "basket_fill", "salary", "salary_index",
             "basket_fill_real", "hours_for_1000", "top3"]]
    ec.to_csv("out/basket_economy_clusters.csv", index=False)
    print(ec[["economy_cluster", "type_name", "n_mo", "cores", "basket_fill_real"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
