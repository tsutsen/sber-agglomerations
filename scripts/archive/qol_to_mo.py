"""МО-уровневые фичи из индекса качества жизни (развивай.рф, 2023-2024).

Соединяет:
  data/qol_directions.csv   (11 направлений × 255 городов × 2 года)
  data/qol_city_mo_map.csv  (город -> МО, usable только)
в wide-таблицу по МО:
  data/mo_qol.csv  (territory_id, год, 11 направлений, ikz8 = среднее
  8 базовых направлений; 3 дополнительные — activity/leisure/satisfaction —
  в отдельные колонки, в ikz8 не входят).

Запуск: PYTHONPATH=src .venv/bin/python src/data/qol_to_mo.py
"""
from __future__ import annotations

import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Та же маппинг-таблица, что в qol_extract.py (2023/2024 + номенклатура 2025).
DIRS_EN = {
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
# 8 сопоставимых семантически направлений (одни и те же ключи во все 3 года —
# ikz8 сравним по годам по построению).
CORE = ["housing", "income", "health", "education",
        "mobility", "environment", "ecology", "safety"]
EXTRA25 = ["work_business", "social_protection", "family"]


def main() -> None:
    d = pd.read_csv(os.path.join(ROOT, "data", "qol_directions.csv"))
    m = pd.read_csv(os.path.join(ROOT, "data", "qol_city_mo_map.csv"))
    m = m[m.usable]
    # коллизии: несколько городов -> одно МО
    dup = m[m.territory_id.duplicated(keep=False)]
    if len(dup):
        print("WARNING cities->MO collisions:")
        print(dup[["nameEng", "territory_id", "municipal_district_name"]].to_string(index=False))
    # Единственная коллизия в датасете: Ивантеевка и Пушкино — оба в
    # ГО Пушкинский. Оставляем первый попавшийся (оба характеризуют одно МО);
    # полные данные обоих городов сохраняются в qol_directions.csv.
    m = m.drop_duplicates("territory_id", keep="first")
    # qol_directions.csv уже содержит англоязычные ключи направлений
    d["value_city"] = pd.to_numeric(d.value_city, errors="coerce")
    d = d.merge(m[["nameEng", "territory_id", "municipal_district_name"]],
                left_on="city", right_on="nameEng")
    wide = d.pivot_table(index=["territory_id", "municipal_district_name", "year"],
                         columns="direction", values="value_city", aggfunc="first")
    wide = wide.reset_index()
    wide["ikz8"] = wide[CORE].mean(axis=1)
    cols = (["territory_id", "municipal_district_name", "year"]
            + list(dict.fromkeys(list(DIRS_EN.values()) + EXTRA25)) + ["ikz8"])
    cols = [c for c in cols if c in wide.columns]
    wide = wide[cols].sort_values(["territory_id", "year"])
    path = os.path.join(ROOT, "data", "mo_qol.csv")
    wide.to_csv(path, index=False)
    print(f"mo_qol: {wide.shape} ({wide.territory_id.nunique()} MOs, "
          f"years {sorted(wide.year.unique())})")
    print(wide[wide.year == 2024].describe().T[["mean", "std", "min", "max"]].to_string())
    # самопроверка: Байкальск 2024
    b = wide[(wide.territory_id == int(m[m.nameEng == "Baikalsk"].territory_id.iloc[0]))
             & (wide.year == 2024)]
    assert abs(b.housing.iloc[0] - 45.468478) < 1e-4, b
    print("self-check Baikalsk housing 2024 OK")


if __name__ == "__main__":
    main()
