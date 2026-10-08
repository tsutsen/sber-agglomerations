"""Картируем «индекс мобильности» Сбера (data/indeks-mobilnosti_ru_*.csv) на наши
territory_id через municipal_district_name и пишем data/mo_mobility.csv
(territory_id, mobility_2024, mobility_2025, км).

Запуск из корня:  python scripts/03_mobility.py
"""
from __future__ import annotations

import glob
import os

import pandas as pd


def norm(s) -> str:
    return " ".join(str(s).lower().split())


def main() -> None:
    src = sorted(glob.glob("data/indeks-mobilnosti_ru_*.csv"))[0]
    d = pd.read_csv(src, sep=";")
    d.columns = [c.strip() for c in d.columns]
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    base = pd.read_csv("data/mo_consumption_population.csv",
                       usecols=["territory_id", "municipal_district_name"])
    name2id = {norm(n): int(t) for t, n in zip(base.territory_id, base.municipal_district_name)}
    d["territory_id"] = d.ref_area.map(lambda x: name2id.get(norm(x)))
    matched = d.dropna(subset=["territory_id"])
    print(f"попало в базу: {matched.territory_id.nunique()} МО из {d.ref_area.nunique()}")
    piv = matched.pivot_table(index="territory_id", columns="period", values="value")
    piv.columns = [c[:4] for c in piv.columns]
    piv = piv.rename(columns={"2024": "mobility_2024", "2025": "mobility_2025"})
    piv.index.name = "territory_id"
    os.makedirs("data", exist_ok=True)
    piv.to_csv("data/mo_mobility.csv")
    print("-> data/mo_mobility.csv")


if __name__ == "__main__":
    main()
