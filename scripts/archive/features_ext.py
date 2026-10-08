"""Расширенное статическое пространство признаков МО (дополнение к 6 динамическим).

Статические признаки (2024, фолбэк 2023), по одному на МО:
  log_salary       лог средней зарплаты (Все отрасли, годовое значение)
  work_share       доля трудоспособного населения (15–64)
  mig_per_1000     чистая миграция на 1000 жителей (годовая)
  log_market_acc   лог индекса доступности рынка (готовая мера центральности)
  cons_growth      рост среднего расхода 2024 к 2023
  season_amp       амплитуда сезонности: max/min месячного расхода за 24 мес
  qol_* (9)        индекс качества жизни (развивай.рф, 2024/фолбэк 2023):
                   8 базовых направлений (0-100) + qol_ikz8 (их среднее),
                   qol_has — флаг наличия данных (только 234 МО-города)

Динамические 6 признаков (5 долей + log-уровень) — из load_features, по месяцам.

Читает:  data/raw/sberindex_hackathon 070625/{2,3,4}_bdmo_*.parquet,
          data/1_market_access.parquet, data/mo_consumption_population.csv,
          data/mo_consumption_monthly_long.csv
Пишет:   data/mo_static_features.csv

Запуск из корня:  .venv/bin/python src/data/features_ext.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW = "data/raw/sberindex_hackathon 070625"
STATIC_COLS = ["log_salary", "work_share", "mig_per_1000",
               "log_market_acc", "cons_growth", "season_amp"]
# Индекс качества жизни (развивай.рф, 2024, фолбэк 2023): 8 базовых
# направлений + ikz8 (их среднее). Есть только для 234 МО (городов
# индекса) — на остальных median-импутация, флаг qol_has.
QOL_COLS = ["qol_housing", "qol_income", "qol_health", "qol_education",
            "qol_mobility", "qol_environment", "qol_ecology", "qol_safety",
            "qol_ikz8"]
QOL_SRC = {"qol_housing": "housing", "qol_income": "income",
           "qol_health": "health", "qol_education": "education",
           "qol_mobility": "mobility", "qol_environment": "environment",
           "qol_ecology": "ecology", "qol_safety": "safety",
           "qol_ikz8": "ikz8"}


def _qol() -> pd.DataFrame:
    q = pd.read_csv("data/mo_qol.csv")
    q = q.sort_values("year").groupby("territory_id", as_index=False).last()
    out = q.set_index("territory_id")[list(QOL_SRC.values())]
    return out.rename(columns={v: k for k, v in QOL_SRC.items()})


def _salary() -> pd.Series:
    s = pd.read_parquet(f"{RAW}/4_bdmo_salary.parquet")
    s = s[(s.okved_name == "Все отрасли") & (s.period == "январь-декабрь")]
    s = s.pivot_table(index="territory_id", columns="year", values="value")
    v = s.get(2024, pd.Series(dtype=float)).fillna(s.get(2023))
    return np.log1p(v.clip(lower=0)).rename("log_salary")


def _work_share() -> pd.Series:
    p = pd.read_parquet(f"{RAW}/2_bdmo_population.parquet")
    a = p.age.astype(str)

    def num(x: str) -> float | None:
        return float(x) if x.isdigit() else None

    ages = a.map(num)
    p["age_num"] = ages
    tot = p[p.age == "Всего"].groupby("territory_id").value.sum()
    work = p[(p.age_num >= 15) & (p.age_num <= 64)].groupby("territory_id").value.sum()
    # clip: в отдельных МО возрастной расклад дублирует старшие группы (1796)
    return (work / tot).clip(0.0, 1.0).rename("work_share")


def _migration() -> pd.Series:
    m = pd.read_parquet(f"{RAW}/3_bdmo_migration.parquet")
    m = m[m.age == "Всего"]
    net = m.groupby("territory_id").value.sum()
    p = pd.read_parquet(f"{RAW}/2_bdmo_population.parquet")
    tot = p[p.age == "Всего"].groupby("territory_id").value.sum()
    return (net / tot * 1000).rename("mig_per_1000")


def main() -> None:
    base = pd.read_csv("data/mo_consumption_population.csv",
                       usecols=["territory_id", "cons_total_2023", "cons_total_2024"]) \
        .set_index("territory_id")
    # index = territory_id: иначе присваивание tid-индексированных Series
    # выровнялось бы по RangeIndex (сдвиг значений на соседние строки)
    out = pd.DataFrame({"territory_id": base.index.to_numpy()}, index=base.index)
    out["log_salary"] = _salary().reindex(base.index)
    out["work_share"] = _work_share().reindex(base.index)
    out["mig_per_1000"] = _migration().reindex(base.index)
    ma = pd.read_parquet(f"{RAW}/1_market_access.parquet").set_index("territory_id")
    out["log_market_acc"] = np.log1p(ma.market_access.reindex(base.index).clip(lower=0))
    g = (base.cons_total_2024 - base.cons_total_2023) / base.cons_total_2023.replace(0, np.nan)
    out["cons_growth"] = g
    # сезонность: max/min месячной суммы по 5 категориям за 24 мес
    m = pd.read_csv("data/mo_consumption_monthly_long.csv",
                    usecols=["territory_id", "date", "consumption_rub"])
    m = m[m.consumption_rub > 0]
    mon = m.groupby(["territory_id", "date"]).consumption_rub.sum()
    amp = (mon.groupby(level=0).max() / mon.groupby(level=0).min()).rename("season_amp")
    out["season_amp"] = amp.reindex(base.index)
    q = _qol().reindex(base.index)
    out[QOL_COLS] = q
    out["qol_has"] = q["qol_housing"].notna().astype(int)

    for c in STATIC_COLS + QOL_COLS:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(out[c].median())
    print(f"накрыто: {out.territory_id.nunique()} МО; NaN после импутации: 0")
    print(out.describe(include=np.number).T[["mean", "std", "min", "max"]].round(2).to_string())
    out.to_csv("data/mo_static_features.csv", index=False)
    print("-> data/mo_static_features.csv")


if __name__ == "__main__":
    main()
