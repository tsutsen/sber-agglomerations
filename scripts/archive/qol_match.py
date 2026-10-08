"""Сопоставление городов индекса качества жизни (развивай.рф) с МО.

Точка города (lat/lon из qol_cities_full.json) -> полигон МО
(data/mo_consumption_population.gpkg, 2548 МО). Контроль: регион города
(с сайта) против региона МО (из gpkg).

Запуск: PYTHONPATH=src .venv/bin/python src/data/qol_match.py
Результат: data/qol_city_mo_map.csv
"""
from __future__ import annotations

import json
import os

import geopandas as gpd
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# На сайте развивай.рф перепутаны координаты двух городов —
# правим известными (Википедия/ФГИС):
_COORD_FIX = {
    "Vologda": (60.6250, 36.3054),
    "Syktyvkar": (61.6687, 50.8337),
}

# Города, у которых собственный городской округ отсутствует в датасете МО
# (2548): точка города падает в соседний район, и данные города нельзя
# считать фичей этого района (проверено визуально по географии).
_NOT_USABLE = {
    "Baikalsk": "город Байкальск нет в датасете, точка в Слюдянском р-не",
    "Vologda": "город Вологда нет в датасете, точка в Вытегорском р-не",
    "Zheleznogorsk-Ilimsky": "г. Железногорск-Илимский нет, точка в Нижнеилимском р-не",
    "Krasnodar": "город Краснодар нет в датасете, точка в Брюховецком р-не",
    "Seversk": "город Северск нет в датасете, точка в Томском р-не",
}


def main() -> None:
    cities = json.load(open(os.path.join(ROOT, "data", "raw", "qol_cities_full.json"),
                            encoding="utf-8"))
    g = gpd.read_file(os.path.join(ROOT, "data", "mo_consumption_population.gpkg"),
                      columns=["territory_id", "municipal_district_name",
                               "municipal_district_type", "region_name"])
    gcrs = g.crs
    for c in cities:
        if c["nameEng"] in _COORD_FIX:
            c["lat"], c["lon"] = _COORD_FIX[c["nameEng"]]
    pt = gpd.GeoDataFrame(cities,
                          geometry=gpd.points_from_xy([c["lon"] for c in cities],
                                                     [c["lat"] for c in cities],
                                                     crs="EPSG:4326").to_crs(gcrs))
    gcols = g[["geometry", "territory_id", "municipal_district_name",
               "municipal_district_type", "region_name"]].assign(
        area=g.geometry.area)
    joined = gpd.sjoin(pt, gcols, how="left", predicate="within")
    out = joined[["nameEng", "name", "region", "territory_id",
                  "municipal_district_name", "municipal_district_type",
                  "region_name", "area"]].rename(columns={"region_name": "mo_region"})
    # 3 города (Березники, Коломна, Орехово-Зуево) лежат и в городском округе,
    # и в прилегающем районе — берём меньший полигон (городской округ).
    out = (out.sort_values("area", na_position="last")
           .drop_duplicates("nameEng", keep="first")
           .sort_index().reset_index(drop=True))
    out["unmatched"] = out.territory_id.isna()
    out["match_method"] = "within"
    # Fallback: ближайший по границе МО (города на воде/берегу, дырки в
    # полигонах). Только для несостыкованных.
    if out["unmatched"].any():
        # географическая СК: расстояние в градусах, ~111 км/град — ок
        # для метки match_method (точность не критична).
        nn = gpd.sjoin_nearest(pt[out["unmatched"]], gcols, how="left",
                               distance_col="dist_deg")
        for idx, row in nn.iterrows():
            mrow = g[g.territory_id == row.territory_id].iloc[0]
            out.loc[idx, ["territory_id", "municipal_district_name",
                          "municipal_district_type", "mo_region"]] = [
                mrow.territory_id, mrow.municipal_district_name,
                mrow.municipal_district_type, mrow.region_name]
            out.loc[idx, "unmatched"] = False
            out.loc[idx, "match_method"] = f"nn_{row.dist_deg * 111:.1f}km"
    # регион сайта vs регион МО: сравним по общему началу (сокращения типа
    # "область"/"обл" не обязаны совпадать дословно)
    out["region_ok"] = out.apply(
        lambda r: (not r.unmatched) and
        (r.region[:6] in r.mo_region or r.mo_region[:6] in r.region
         or r.region[:4] == r.mo_region[:4]), axis=1)
    # usable: город реально лежит в этом МО. nn-состыковка к ЧУЖОМУ району
    # (Улан-Удэ/Южно-Сахалинск/Элиста и т.д. — их ГО нет в датасете) не
    # пригодна для фич этого МО. Moscow -> Тверской (внутригородской) —
    # данные по всему городу, использовать только агрегированно.
    def _usable(r):
        if r.nameEng == "Moscow":
            return False
        if r.match_method == "within":
            return True
        # nn: годится, если имя города встречается в названии МО
        return r["name"] in r["municipal_district_name"]
    out["usable"] = out.apply(_usable, axis=1)
    out["note"] = ""
    out.loc[out.nameEng == "Moscow", "note"] = "данные по всему городу, не фича МО"
    out.loc[out.nameEng == "Sevastopol", "note"] = "Севастополя в датасете нет"
    out.loc[out.nameEng.isin(_NOT_USABLE), "usable"] = False
    out.loc[out.nameEng.isin(_NOT_USABLE), "note"] = out.nameEng.map(_NOT_USABLE)
    out.loc[out.nameEng == "Moscow", "usable"] = False
    path = os.path.join(ROOT, "data", "qol_city_mo_map.csv")
    out.to_csv(path, index=False)
    print(f"cities={len(out)} unmatched={int(out.unmatched.sum())} "
          f"region_mismatch={int((~out.region_ok & ~out.unmatched).sum())}")
    print("\nUnmatched cities:")
    print(out[out.unmatched][["nameEng", "name", "region"]].to_string(index=False))
    print("\nRegion mismatches:")
    print(out[~out.region_ok & ~out.unmatched]
          [["nameEng", "region", "municipal_district_name", "mo_region"]]
          .to_string(index=False))
    print("\nMO type distribution:")
    print(out.municipal_district_type.value_counts().to_string())
    # какие города попали в тип МО «городской округ» vs «муниципальный район»
    print("\nSample matches:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
