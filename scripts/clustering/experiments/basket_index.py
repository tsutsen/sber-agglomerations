"""Индекс тележки: одна стандартная корзина, одни и те же 1000 руб., разные регионы.

Два слоя (масштаб общий, среднероссийский = 100%):
  fill_price = 100 / coef24          — что покупают 1000 руб. при местной цене
  fill_real  = fill_price × (зарплата_регион / зарплата_РФ)  — та же корзина
               на среднюю местную зарплату (real wage index)

Вывод: out/basket_index_regions.csv, out/basket_index_types.csv,
out/basket_index_mo.csv, out/basket_index_summary.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

MON = {n: i + 1 for i, n in enumerate(
    ["январь", "февраль", "март", "апрель", "май", "июнь",
     "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"])}
TYP = "out/types_profile_g3.csv"
WORKH = 173  # штатных рабочих часов в месяце (21,75 дн x 8 ч)


def main() -> None:
    # --- цена: средний годовой коэффициент корзины по регионам (2024)
    b = pd.read_csv("data/sources/basket_size_per_mo.csv")
    b["t"] = (b.year - 2023) * 12 + b.month.map(MON) - 1
    coef = b.pivot_table(index="region_name", columns="t", values="value")
    coef24 = coef.iloc[:, 12:24].mean(axis=1)

    # --- МО: регион + тип + зарплата
    import geopandas as gpd
    g = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                    usecols=["territory_id", "region_name", "municipal_district_name"]) \
        .drop_duplicates("territory_id")
    lng = pd.read_csv("data/sources/derived/mo_consumption_monthly_long.csv", usecols=["territory_id"])
    work = set(lng.territory_id.unique())
    g = g[g.territory_id.isin(work)]

    g = g.set_index("territory_id")
    from ..features import FEAT_COLS, load_features
    f2, _ = load_features()
    dyn = f2.groupby("mo")[FEAT_COLS].mean()
    static = pd.read_csv("data/experiments/inputs/derived/mo_static_features.csv").set_index("territory_id")
    g["salary"] = np.expm1(static.log_salary.reindex(g.index).fillna(0))
    STATIC = ["log_salary", "work_share", "mig_per_1000", "log_market_acc",
              "cons_growth", "season_amp"]
    # log_basket / growth_basket — как в features_exp4: по матрице (MO x 24 мес)
    bb = pd.read_csv("data/sources/basket_size_per_mo.csv")
    bb["t"] = (bb.year - 2023) * 12 + bb.month.map(MON) - 1
    bwide = bb.pivot_table(index="region_name", columns="t", values="value")
    bmo = pd.concat([g.region_name.map(bwide[t]).rename(int(t)) for t in bwide.columns],
                    axis=1).reindex(g.index)  # (27, MO); NaN — регионы без данных
    bmo = bmo.fillna(bmo.median(axis=1))
    sb = static.copy()
    sb["log_basket"] = np.log1p(bmo.mean(axis=0))
    sb["growth_basket"] = (bmo.iloc[13:25].mean(axis=0) /
                           bmo.iloc[1:13].mean(axis=0).clip(lower=1e-9) - 1)
    stat8 = sb[STATIC + ["log_basket", "growth_basket"]]

    ids = np.sort(f2.mo.unique())
    Xd = dyn.reindex(ids).fillna(0.0).to_numpy(float)
    Xs = stat8.reindex(ids).fillna(0.0).to_numpy(float)
    X = np.hstack([Xd, Xs])
    X = (X - X.mean(0)) / X.std(0, ddof=0)
    g2 = g.reindex(ids)

    tp = pd.read_csv(TYP)
    ax = ["share_food", "share_health", "share_cafe", "share_transport",
          "share_marketplace", "log_total"] + \
         ["log_salary", "work_share", "mig_per_1000", "log_market_acc",
          "cons_growth", "season_amp", "log_basket", "growth_basket"]
    cents = tp[ax].to_numpy(float)
    d2 = ((X[:, None, :] - cents[None, :, :])**2).sum(-1)
    ttype = np.argmin(d2, axis=1)  # индекс в строках tp
    g2["type"] = tp["cluster"].to_numpy()[ttype]  # id типа (0..16)
    g2["cores"] = tp["cores"].to_numpy()[ttype]

    # --- сборка таблиц
    g2["coef24"] = g2.region_name.map(coef24)
    g2["fill_price"] = (100.0 / g2.coef24).round(1)  # % стандартной корзины за 1000 руб.
    sal_rf = float(g2.salary.mean())
    g2["salary_index"] = (100.0 * g2.salary / sal_rf).round(1)
    g2["fill_real"] = (g2.fill_price * g2.salary_index / 100.0).round(1)
    g2["hours_for_1000"] = (1000.0 * WORKH / g2.salary).round(2)  # часов средней работы на 1000 руб.

    mo_cols = ["region_name", "municipal_district_name", "type",
               "cores", "coef24", "fill_price", "salary", "salary_index",
               "fill_real", "hours_for_1000"]
    out_mo = g2[mo_cols].copy()
    out_mo.insert(0, "territory_id", out_mo.index)
    out_mo.to_csv("out/basket_index_mo.csv", index=False)

    reg = g2.groupby("region_name").agg(
        coef24=("coef24", "mean"), salary=("salary", "mean"),
        fill_price=("fill_price", "mean"), salary_index=("salary_index", "mean"),
        fill_real=("fill_real", "mean"), hours_for_1000=("hours_for_1000", "mean"),
        n_mo=("type", "size")).reset_index()
    reg.to_csv("out/basket_index_regions.csv", index=False)

    cores = tp.set_index("cluster")["cores"].rename("cores_type")
    agg = g2.groupby("type").agg(
        coef24=("coef24", "mean"), salary=("salary", "mean"),
        fill_price=("fill_price", "mean"), salary_index=("salary_index", "mean"),
        fill_real=("fill_real", "mean"), hours_for_1000=("hours_for_1000", "mean"),
        n_mo=("type", "size"))
    types = agg.join(cores).reset_index()
    for c in ["coef24", "salary_index", "fill_price", "fill_real", "hours_for_1000"]:
        types[c] = types[c].round(1)
    types["salary"] = types.salary.round(0)
    types = types.sort_values("fill_real")
    types.to_csv("out/basket_index_types.csv", index=False)

    r = reg.set_index("region_name")
    summary = {
        "scale": "все показатели в %, среднероссийский = 100; одна стандартная корзина, одинаковые деньги",
        "fill_price": "100 / коэффициент корзины-2024 (что дают 1000 руб.)",
        "fill_real": "fill_price * средняя зарплата региона / средняя зарплата РФ (та же корзина на местную зарплату)",
        "hours_for_1000": "часов средней местной работы на 1000 руб. (зарплата месячная, 173 ч/мес)",
        "salary_units": "руб/мес, средняя по МО (год 2024, все отрасли), агрегат = среднее МО (не взвешено)",
        "n_regions": int(reg.coef24.notna().sum()),
        "n_mo": int(len(g2)),
        "cheapest": {"region": str(r.fill_price.idxmax()),
                     "fill_price": float(r.fill_price.max()),
                     "fill_real": float(r.loc[r.fill_price.idxmax(), "fill_real"])},
        "most_expensive": {"region": str(r.fill_price.idxmin()),
                           "fill_price": float(r.fill_price.min()),
                           "fill_real": float(r.loc[r.fill_price.idxmin(), "fill_real"])},
        "caveats": [
            "коэффициент корзины — регионального уровня (внутри региона все МО равны)",
            "зарплата — средняя месячная (2024, все отрасли); агрегаты не взвешены по численности; медиана неизвестна",
            "стандартная корзина — корзина хакатона (27 мес. данных), не весь потребительскийbasket",
            "4 региона без коэффициента — NaN",
            "это факторный (counterfactual) расчёт: одинаковый покупатель, разные цены/зарплаты",
        ],
    }
    json.dump(summary, open("out/basket_index_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"зарплата РФ (ср. МО): {sal_rf:,.0f} руб/мес")
    print("\nРегионы (fill_price, fill_real):")
    print(reg.sort_values("fill_real")[
        ["region_name", "coef24", "fill_price", "salary", "fill_real",
         "hours_for_1000"]].to_string(index=False))
    print("\nТипы (14 осей):")
    print(types[["type", "n_mo", "cores_type", "coef24", "fill_price",
                 "salary_index", "fill_real", "hours_for_1000"]].to_string(index=False))


if __name__ == "__main__":
    main()
