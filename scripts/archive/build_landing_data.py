#!/usr/bin/env python3
"""Собирает GeoJSON-слои для landing-карты:

  data/clusters.geojson         — МО с типом/цветом кластера (клип по МО, ~100 м)
  data/cluster_outlines.geojson — сшитые контуры кластеров (только внешние кольца)
  data/official.geojson         — сшитые контуры официального перечня агломераций
                                  (только внешние кольца, флаг scattered)

Запуск из корня проекта:  .venv/bin/python src/viz/build_landing_data.py
"""
from __future__ import annotations

import json
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon


def _exists(p: str) -> bool:
    return os.path.exists(p)


def fill_holes(geom):
    """Только внешние кольца — кластер выглядит монолитным."""
    if geom is None:
        return geom
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    if isinstance(geom, MultiPolygon):
        polys = [Polygon(p.exterior) for p in geom.geoms]
        return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    return geom


def dump(gdf: gpd.GeoDataFrame, path: str) -> int:
    # Дальше НЕ упрощаем: контуры уже общие (клип по МО) — повторное
    # упрощение по отдельности снова содрогнет швы. Дырки не заливаем:
    # они либо анклав чужого кластера (477 окружает Куркино из 864),
    # либо МО без данных — оба случая честные, а заливка создаёт перекрешение.
    gdf = gdf.copy()
    gj = json.loads(gdf.to_json())  # type: ignore[arg-type]
    gj["type"] = "FeatureCollection"
    with open(path, "w") as f:  # type: ignore[unreachable]
        json.dump(gj, f, ensure_ascii=False)
    return len(gj["features"])


def main() -> None:
    g = gpd.read_file("data/mo_consumption_population.gpkg")
    cl = pd.read_csv("out/clusters_all.csv")
    final = cl[cl.window == cl.window.max()].drop_duplicates("mo").rename(columns={"mo": "territory_id"})  # type: ignore[call-overload, call-overload]
    rep = pd.read_csv("out/cluster_vs_official.csv")
    m = g.merge(final, on="territory_id", how="inner")

    m["type"] = m.cluster.map(rep.set_index("cluster").type.fillna("small"))
    m["n_mo"] = m.cluster.map(rep.set_index("cluster").n_mo)
    m["dominant"] = m.cluster.map(rep.set_index("cluster").dominant_agg.fillna(""))
    m["color"] = "#bbbbbb"
    for t, c in (("match", "#2ca02c"), ("match_partial", "#ffbb78"), ("match_expanded", "#17becf"),
                 ("merge", "#1f77b4"), ("new_market", "#e41a1c")):
        m.loc[m.type == t, "color"] = c
    gdf_mo = gpd.GeoDataFrame(
        m[["territory_id", "municipal_district_name", "region_name", "cluster",
           "type", "n_mo", "dominant", "color"]].rename(columns={"n_mo": "cluster_size"}),
        geometry=m.geometry, crs=m.crs)

    # size = средние расходы; activity = расходы × население (эконом. масса МО)
    size = pd.read_csv("data/mo_consumption_population.csv")[
        ["territory_id", "cons_total_2024", "pop_2024"]].copy()
    size["size"] = size.cons_total_2024.fillna(0)
    size["activity"] = size.cons_total_2024.fillna(0) * size.pop_2024.fillna(0)
    gdf_s = gdf_mo.merge(size[["territory_id", "activity"]], on="territory_id", how="left")  # type: ignore[arg-type]

    # Имена: словарь data/cluster_names.csv (все кластеры, выведены из топ-3
    # по активности). Core — топ-3 МО [[имя, активность ₽/мес], ...].
    names = pd.read_csv("data/cluster_names.csv").set_index("cluster") if _exists("data/cluster_names.csv") else None
    name_by, core_by = {}, {}
    for c, grp in gdf_s.groupby("cluster"):
        top = grp.sort_values("activity", ascending=False).head(3)
        core_by[c] = [[r.municipal_district_name, int(r.activity)] for r in top.itertuples() if pd.notna(r.activity)]
        name_by[c] = names.loc[c, "name"] if names is not None and c in names.index else top.iloc[0].municipal_district_name

    # Исходные МО местами слегка перекрывают друг друга. Клипшим на уровне МО
    # (упрощённые), потом сливаем: соседние кластеры делят ОДИН и тот же обрезанный
    # контур — швы совпадают, дырок и двойных слоёв нет.
    work = gdf_mo.copy()
    work["geometry"] = work.geometry.simplify(0.001, preserve_topology=True)  # ~100 m: общее для всех МО
    b = work.geometry.bounds
    extent = (b.maxx - b.minx) * (b.maxy - b.miny)
    work = work.reindex(extent.sort_values(ascending=False).index)
    done = None
    clip_list = []
    for geom in work.geometry:
        g = geom if done is None else geom.difference(done)
        clip_list.append(g)
        done = g if done is None else done.union(g)
    work["geometry"] = clip_list
    out = work.dissolve(by="cluster").reset_index()
    agg = gdf_mo.groupby("cluster").agg(type=("type", "first"), color=("color", "first"))
    out["type"] = out.cluster.map(agg["type"])
    out["color"] = out.cluster.map(agg["color"])
    out["name"] = [name_by[c] for c in out.cluster]
    out["core"] = [core_by[c] for c in out.cluster]
    # Полный список МО кластера (по убыванию активности) — для popup
    mo_list = (gdf_s.sort_values("activity", ascending=False)
               .groupby("cluster").municipal_district_name.apply(list).to_dict())
    out["mo_list"] = [mo_list[c] for c in out.cluster]
    # Внутренние кольца (дырки-анклавы чужих кластеров) видимы как «линии внутри» —
    # оставляем только внешние контуры.
    out["geometry"] = out.geometry.apply(fill_holes)
    n1 = dump(out, "data/cluster_outlines.geojson")  # type: ignore[arg-type]

    # МО-слой — те же обрезанные геометрии: швы МО == швы кластеров
    n0 = len(work)
    work.to_file("data/clusters.geojson", driver="GeoJSON")

    off = pd.read_csv("data/official_agglomerations_mapped.csv")
    cov = pd.read_csv("out/official_coverage.csv").set_index("agglomeration")
    o = work.merge(off[["territory_id", "Агломерация"]].dropna(subset=["territory_id"]),  # type: ignore[call-overload]
                     on="territory_id", how="inner").rename(columns={"Агломерация": "agg"})  # type: ignore[call-overload]
    d = o.dissolve(by="agg").reset_index()
    d["scattered"] = [bool(cov.loc[a, "scattered"]) if a in cov.index else False for a in d["agg"]]
    off_list = (o.sort_values("activity", ascending=False).groupby("agg")
                .municipal_district_name.apply(list).to_dict())
    d["mo_list"] = [off_list[a] for a in d["agg"]]
    # Официальный слой рисуется только контуром — внутренние кольца (дырки от
    # анклавов чужих кластеров) видны как «линии внутри polygons», убираем их.
    d["geometry"] = d.geometry.apply(fill_holes)
    n2 = dump(d, "data/official.geojson")
    print(f"clusters.geojson: {n0} МО | outlines: {n1} | official: {n2}")


if __name__ == "__main__":
    main()
