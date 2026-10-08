"""Сборка: потребление СберИндекса (5 категорий + все) + население Росстата + геометрия МО."""
import glob
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

# GDAL трактует «!» в абсолютном пути как разделитель архива — работаем с относительными путями
os.chdir(Path(__file__).resolve().parent)
ROOT = Path(".")
SRC = Path("..") / "данные сбериндекс" / "sberindex_hackathon 070625"
OUT_GPKG = ROOT / "mo_consumption_population.gpkg"
OUT_CSV = ROOT / "mo_consumption_population.csv"
OUT_MONTHLY = ROOT / "mo_consumption_monthly_long.csv"

CATEGORIES = {
    "Продовольствие": "food",
    "Здоровье": "health",
    "Общественное питание": "cafe",
    "Транспорт": "transport",
    "Маркетплейсы": "marketplace",
    "Все категории": "total",
}

# --- население: строки «Всего», сумма по полам -------------------------------
pop = pd.read_parquet(SRC / "2_bdmo_population.parquet").drop_duplicates()
pop = pop[pop.age == "Всего"]
pop = (
    pop.groupby(["territory_id", "year"])
    .agg(population=("value", "sum"), n_gender=("gender", "nunique"))
    .reset_index()
)
pop.loc[pop.n_gender < 2, "population"] = pd.NA
pop_wide = pop.pivot(index="territory_id", columns="year", values="population")
pop_wide.columns = [f"pop_{y}" for y in pop_wide.columns]

# --- потребление -------------------------------------------------------------
cons = pd.read_parquet(SRC / "8_consumption.parquet")
cons["territory_id"] = cons.territory_id.astype(int)
cons["year"] = cons.date.str[:4].astype(int)
cons["cat"] = cons.category.map(CATEGORIES)

annual = (
    cons.groupby(["territory_id", "cat", "year"])
    .agg(mean=("value", "mean"), months=("value", "size"))
    .reset_index()
)
cons_wide = annual.pivot(index="territory_id", columns=["cat", "year"], values="mean").round(0)
cons_wide.columns = [f"cons_{c}_{y}" for c, y in cons_wide.columns]
months = (
    annual[annual.cat == "total"]
    .pivot(index="territory_id", columns="year", values="months")
    .add_prefix("cons_months_")
)
order = [f"cons_{c}_{y}" for c in CATEGORIES.values() for y in (2023, 2024)]
cons_wide = cons_wide[order].join(months)

# --- справочник и геометрия --------------------------------------------------
gpkg_path = glob.glob("t_dict_municipal*/*.gpkg")[0]
xlsx_path = glob.glob("t_dict_municipal*/*.xlsx")[0]

geom = gpd.read_file(gpkg_path)
geom["territory_id"] = geom.territory_id.astype(int)
geom = geom[["territory_id", "year_from", "year_to", "geometry"]]

dic = pd.read_excel(xlsx_path)
dic = dic.sort_values("year_to").drop_duplicates("territory_id", keep="last")
dic = dic[[
    "territory_id", "oktmo", "municipal_district_name", "municipal_district_name_short",
    "municipal_district_type", "region_code", "region_name",
    "municipal_district_center_lat", "municipal_district_center_lon",
]]

ids = cons_wide.index.union(pop_wide.index)
gdf = (
    geom[geom.territory_id.isin(ids)]
    .merge(dic, on="territory_id", how="left")
    .merge(pop_wide, left_on="territory_id", right_index=True, how="left")
    .merge(cons_wide, left_on="territory_id", right_index=True, how="left")
)
gdf["has_consumption"] = gdf.cons_total_2024.notna() | gdf.cons_total_2023.notna()
gdf = gdf.sort_values("territory_id")
gdf = gdf[[c for c in gdf.columns if c != "geometry"] + ["geometry"]]

gdf.to_file(OUT_GPKG, layer="mo_consumption_population", driver="GPKG")
gdf.drop(columns="geometry").to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

# --- помесячная длинная таблица ----------------------------------------------
monthly = (
    cons.merge(pop[["territory_id", "year", "population"]], on=["territory_id", "year"], how="left")
    .merge(dic[["territory_id", "municipal_district_name", "region_name"]], on="territory_id", how="left")
    [["territory_id", "municipal_district_name", "region_name", "date", "year",
      "category", "cat", "value", "population"]]
    .rename(columns={"value": "consumption_rub"})
    .sort_values(["territory_id", "date", "cat"])
)
monthly.to_csv(OUT_MONTHLY, index=False, encoding="utf-8-sig")

# --- отчёт -------------------------------------------------------------------
print("МО в результате:", len(gdf))
print("  с потреблением:", gdf.has_consumption.sum(), "| с населением 2023/2024:",
      gdf.pop_2023.notna().sum(), "/", gdf.pop_2024.notna().sum())
print("  id без геометрии:", len(set(ids) - set(geom.territory_id)))
print("  геометрия неактуальна (year_to<2023):", (gdf.year_to < 2023).sum())
print("Помесячных строк:", len(monthly))
for p in (OUT_GPKG, OUT_CSV, OUT_MONTHLY):
    print(p.name, f"{p.stat().st_size / 1e6:.1f} MB")
