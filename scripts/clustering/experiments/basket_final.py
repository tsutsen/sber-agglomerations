"""Финальный пакет данных «Индекс тележки»: одна итоговая таблица + таблицы-источники,
из которых она собирается.

Файлы (out/):
  basket_final.csv            — 2144 МО, все ключи и метрики в одном месте
  mo_salary.csv               — зарплата по МО (мес., 2024, все отрасли)
  mo_expenses.csv             — расходы по МО (год 2024, на человека и всего)
  mo_regions.csv              — регион по МО
  mo_spatial_clusters.csv     — пространственный кластер (production, финальное окно)
  mo_economy_clusters.csv     — тип G3 по МО
  basket_fill_mo.csv          — тележка по МО (price_index_2024, fill, fill_real, hours)
  basket_fill_regions.csv     — fill по регионам (77)
  basket_economy_clusters.csv — характеристики 17 типов (профиль z + fill + ядра)
"""
from __future__ import annotations

import pandas as pd


# Названия 17 типов G3 (final run, types_profile_g3.csv); id = cluster
# Имена по экономическим атрибутам (не по географии)
TYPE_NAMES = {
    16: "Мегаполис: сервисное потребление",
    3:  "Пригороды мегаполисов",
    12: "Добывающий север с высокими зарплатами",
    2:  "Быстрый рост трат",
    6:  "Отрезанные, но высокооплачиваемые",
    13: "Растущие траты, ускоряющиеся цены",
    9:  "Середнячки со скромными расходами",
    1:  "Маркетплейс-провинция",
    10: "Стареющие с низкой инфляцией",
    4:  "Стареющие и изолированные",
    5:  "Дёшевые и скромные",
    8:  "Низкие зарплаты, дешёвые цены",
    0:  "Северный Кавказ",
    11: "Стареющие с низкими зарплатами",
    14: "Быстрая инфляция и экономия на транспорте",
    15: "Дорогая корзина, быстрая инфляция",
    7:  "Дальневосточная периферия",
}

def main() -> None:
    mo = pd.read_csv("out/basket_index_mo.csv").rename(
        columns={"fill_real": "basket_fill_real", "coef24": "price_index_2024"})

    # --- названия МО
    g = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                    usecols=["territory_id", "municipal_district_name_short"])
    names = g.drop_duplicates("territory_id").set_index("territory_id")
    mo["municipality"] = mo.territory_id.map(names.municipal_district_name_short)

    # --- итоговая таблица
    final = mo.rename(columns={
        "municipal_district_name": "municipality_full",
        "type": "economy_cluster",
        "fill_price": "basket_fill",
    })[["territory_id", "municipality", "municipality_full", "region_name",
        "economy_cluster", "cores", "price_index_2024", "basket_fill", "salary",
        "salary_index", "basket_fill_real", "hours_for_1000"]].copy()
    final = final.rename(columns={"region_name": "region"})

    ca = pd.read_csv("out/clusters_all.csv")
    w = ca["window"].max()
    sp = ca[ca.window == w][["mo", "cluster"]].rename(
        columns={"mo": "territory_id", "cluster": "spatial_cluster"})
    final = final.merge(sp, on="territory_id", how="left")
    final["spatial_cluster"] = final.spatial_cluster.astype("Int64")
    final["type_name"] = final.economy_cluster.map(TYPE_NAMES)
    final = final[["municipality", "region", "spatial_cluster", "economy_cluster",
                   "type_name", "cores", "salary", "salary_index", "hours_for_1000",
                   "price_index_2024", "basket_fill", "basket_fill_real",
                   "municipality_full", "territory_id"]]
    final.to_csv("out/basket_final.csv", index=False)

    # --- источники
    sal = mo[["territory_id", "municipality", "region_name", "salary",
              "salary_index"]].rename(columns={"region_name": "region"})
    sal["salary_note"] = ("средняя зарплата, руб/мес (2024, все отрасли); "
                          "salary_index = 100 * зарплата / средняя по МО РФ")
    sal.to_csv("out/mo_salary.csv", index=False)

    exp = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                      usecols=["territory_id", "municipal_district_name_short",
                               "region_name", "pop_2024",
                               "cons_food_2024", "cons_health_2024",
                               "cons_cafe_2024", "cons_transport_2024",
                               "cons_marketplace_2024", "cons_total_2024"])
    exp["cons_total_2024_all_rub"] = (
        exp.cons_total_2024.fillna(0) * exp.pop_2024.fillna(0)).round(0)
    exp["units"] = "руб/чел/год (per-capita columns) и руб/год (all)"
    exp = exp.rename(columns={"municipal_district_name_short": "municipality",
                              "region_name": "region", "pop_2024": "population_2024"})
    exp.to_csv("out/mo_expenses.csv", index=False)

    reg = pd.read_csv("out/basket_index_regions.csv").rename(
        columns={"region_name": "region", "coef24": "price_index_2024"}
    )[["region", "price_index_2024", "fill_price", "salary", "salary_index", "fill_real",
       "hours_for_1000", "n_mo"]].rename(columns={
        "fill_price": "basket_fill", "fill_real": "basket_fill_real"})
    reg.to_csv("out/basket_fill_regions.csv", index=False)

    tp = pd.read_csv("out/types_profile_g3.csv")
    b = pd.read_csv("out/basket_index_types.csv").rename(columns={
        "fill_price": "basket_fill", "fill_real": "basket_fill_real",
        "coef24": "price_index_2024"})
    b = b.drop(columns=["n_mo"])
    ec = tp.merge(b, left_on="cluster", right_on="type")
    ec["economy_cluster"] = ec.cluster
    ec["type_name"] = ec.economy_cluster.map(TYPE_NAMES)
    ec["n_regions"] = mo.groupby("type").region_name.nunique().reindex(ec.cluster).values
    ec = ec[["economy_cluster", "type_name", "n_mo", "n_regions", "cores",
             "log_total", "share_food", "share_health", "share_cafe",
             "share_transport", "share_marketplace", "log_salary", "work_share",
             "mig_per_1000", "log_market_acc", "cons_growth", "season_amp",
             "log_basket", "growth_basket", "price_index_2024", "basket_fill",
             "salary", "salary_index", "basket_fill_real", "hours_for_1000",
             "top3"]]
    ec = ec.sort_values("basket_fill_real")
    ec.to_csv("out/basket_economy_clusters.csv", index=False)

    sp2 = sp.rename(columns={"spatial_cluster": "spatial_cluster"})
    sp2.to_csv("out/mo_spatial_clusters.csv", index=False)

    mo["basket_fill"] = mo.fill_price
    mof = mo[mo.basket_fill_real.notna()]
    mof[["territory_id", "municipality", "price_index_2024", "basket_fill",
         "basket_fill_real", "hours_for_1000"]].to_csv(
        "out/basket_fill_mo.csv", index=False)
    mo[["territory_id", "municipality", "region_name"]].rename(
        columns={"region_name": "region"}).to_csv(
        "out/mo_regions.csv", index=False)
    ec_mo = mo[["territory_id", "municipality", "type", "cores"]].rename(
        columns={"type": "economy_cluster"})
    ec_mo["type_name"] = ec_mo.economy_cluster.map(TYPE_NAMES)
    ec_mo[["territory_id", "municipality", "economy_cluster", "type_name",
           "cores"]].to_csv("out/mo_economy_clusters.csv", index=False)

    print("basket_final:", len(final), "строк,", len(final.columns), "колонк")
    print(final[["municipality", "region", "spatial_cluster", "economy_cluster",
                 "salary", "salary_index", "hours_for_1000", "basket_fill",
                 "basket_fill_real"]].head(3).to_string())
    print("\nколонки:", final.columns.tolist())


if __name__ == "__main__":
    main()
