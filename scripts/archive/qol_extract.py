"""Разбор XLSX индекса качества жизни (развивай.рф) в tidy-таблицы.

Исходник: data/raw/qol_cities/<City>.xlsx — по одному XLSX на город,
листы по годам (2015-2026). Берём 2023-2025: у 2025 другая
номенклатура (12 направлений) — сопоставимые направления сводим в те же
английские ключи (income <- "Материальное благополучие" и т.д., см. DIRS_EN);
несопоставимые 2025-направления идут отдельными ключами.

Уровни:
  направление — «Значение направления, город/кластер/среднее по индексу» (0-100)
  показатель  — «Значение показателя, город/...», флаг «Участвует в расчете ИКЖ»

Запуск: PYTHONPATH=src .venv/bin/python src/data/qol_extract.py
Результат: data/qol_directions.csv, data/qol_indicators.csv
"""
from __future__ import annotations

import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "raw", "qol_cities")
YEARS = ("2023", "2024", "2025")

DIRS_EN = {
    # 2023/2024 (11 направлений)
    "Жилищные условия": "housing",
    "Доход и работа": "income",
    "Здоровье": "health",
    "Образование": "education",
    "Мобильность": "mobility",
    "Благоустройство": "environment",
    "Природно-экологические условия": "ecology",
    "Безопасность": "safety",
    "Общественная активность и права граждан": "activity",
    "Проведение свободного времени": "leisure",
    "Удовлетворенность": "satisfaction",
    # 2025 (12 направлений, новая номенклатура)
    "Работа и свое дело": "work_business",
    "Материальное благополучие": "income",
    "Укрепление и охрана здоровья": "health",
    "Образование и развитие": "education",
    "Транспорт": "mobility",
    "Экология": "ecology",
    "Социальная защита": "social_protection",
    "Спорт, культура и досуг": "leisure",
    "Семья (народосбережение)": "family",
}


def main() -> None:
    dirs, inds = [], []
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".xlsx"))
    print(f"{len(files)} files")
    for f in files:
        slug = f[:-5]
        x = pd.ExcelFile(os.path.join(SRC, f))
        for year in YEARS:
            if year not in x.sheet_names:
                continue
            df = x.parse(year)
            if df.empty:
                continue
            # направление: строки со значением направления, без показателя
            d = df[df["Значение направления, город"].notna()
                   & df["Показатель"].isna()].copy()
            d["city"] = slug
            d["year"] = int(year)
            d["direction"] = d["Направление"].map(DIRS_EN)
            d = d[d.direction.notna()]
            d = d.rename(columns={"Значение направления, город": "value_city",
                                  "Значение направления, кластер": "value_cluster",
                                  "Значение направления, среднее по индексу":
                                      "value_index_avg"}).drop(columns="Направление")
            dirs.append(d[["city", "year", "direction", "value_city",
                           "value_cluster", "value_index_avg"]])
            # показатель
            p = df[df["Значение показателя, город"].notna()].copy()
            p["city"] = slug
            p["year"] = int(year)
            # направление наследуем forward-fill'ом (в строках показателя пустое)
            p["direction"] = df["Направление"].ffill()
            p = p.rename(columns={"Показатель": "indicator",
                                  "Участвует в расчете ИКЖ": "in_ikz",
                                  "Относится к ОЭСР": "oecd",
                                  "Eд.изм": "unit",
                                  "Значение показателя, город": "value_city",
                                  "Значение показателя, кластер": "value_cluster",
                                  "Значение показателя, среднее по индексу":
                                      "value_index_avg"})
            inds.append(p[["city", "year", "direction", "indicator", "in_ikz",
                           "oecd", "unit", "value_city", "value_cluster",
                           "value_index_avg"]])
    d = pd.concat(dirs, ignore_index=True)
    p = pd.concat(inds, ignore_index=True)
    for col in ("value_city", "value_cluster", "value_index_avg"):
        p[col] = pd.to_numeric(
            p[col].astype(str).str.replace(",", "."), errors="coerce")
    d.to_csv(os.path.join(ROOT, "data", "qol_directions.csv"), index=False)
    p.to_csv(os.path.join(ROOT, "data", "qol_indicators.csv"), index=False)
    print(f"directions: {d.shape} ({d.city.nunique()} cities)")
    print(f"indicators: {p.shape} ({p.indicator.nunique()} unique names)")
    print("directions 2024:", sorted(d[d.year == 2024].direction.unique()))


if __name__ == "__main__":
    main()
