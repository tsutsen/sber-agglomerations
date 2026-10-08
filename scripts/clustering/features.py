"""Фичи: профиль потребления МО по месяцам.

Вектор МО за месяц: 5 долей категорий (share) + логарифм общего расхода (уровень).
Сырые значения сглаживаются скользящим средним 3 месяца (шум месячных выбросов).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATS = ["food", "health", "cafe", "transport", "marketplace"]
FEAT_COLS = [f"share_{c}" for c in CATS] + ["log_total"]  # вектор профиля (без id-колоннок)
SMOOTH = 3  # мес.


def load_features(data_dir: str = "data/sources/derived") -> tuple[pd.DataFrame, list[str]]:
    """Возвращает (feats, months).

    feats: колонки [mo, t, share_{cat} x5, log_total, total], t = 0..23.
    Оставлены МО с >= 10 месяцами данных (фильтр качества).
    """
    m = pd.read_csv(
        f"{data_dir}/mo_consumption_monthly_long.csv",
        usecols=["territory_id", "date", "cat", "consumption_rub"],
    )
    m = m[m.cat.isin(CATS)]
    m["consumption_rub"] = pd.to_numeric(m.consumption_rub, errors="coerce")

    p = m.pivot_table(
        index=["territory_id", "date"], columns="cat", values="consumption_rub", aggfunc="mean"
    ).reset_index()
    months = sorted(p.date.unique().tolist())
    tmap = {d: i for i, d in enumerate(months)}
    p["t"] = p.date.map(tmap)

    for c in CATS:
        p[c] = p[c].fillna(0.0)  # отсутствующая категория = 0 расход (только в ячейке, не всю строку)
    p = p.sort_values(["territory_id", "t"])
    # сглаживание сырых значений по времени
    for c in CATS:
        p[f"s_{c}"] = (
            p.groupby("territory_id")[c].transform(lambda s: s.rolling(SMOOTH, min_periods=1).mean())
        )
    p["s_total"] = p[[f"s_{c}" for c in CATS]].sum(axis=1)

    good = (
        p.groupby("territory_id").t.nunique()
    )
    keep = good[good >= 10].index
    p = p[p.territory_id.isin(keep)]

    feats = p[["territory_id", "t"]].rename(columns={"territory_id": "mo"})
    for c in CATS:
        feats[f"share_{c}"] = p[f"s_{c}"] / p.s_total
    feats["log_total"] = np.log1p(p.s_total)
    feats["total"] = p.s_total
    return feats.reset_index(drop=True), months
