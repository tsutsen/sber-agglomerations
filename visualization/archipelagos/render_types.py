#!/usr/bin/env python3
"""Рисует картинки для лендинга: обзорную карту и по примеру на каждый тип кластера.

Запуск:  MPLCONFIGDIR=/tmp/mpl python3 render_types.py   → img/*.png
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).parent
OUT = HERE / "img"
OUT.mkdir(exist_ok=True)

plt.rcParams["font.family"] = ["Helvetica Neue", "Arial Unicode MS", "sans-serif"]

BRAND = "#6139E7"
INK = "#1d2433"
LAND = "#E9E6F2"
MO_EDGE = "#FFFFFF"
REGION_EDGE = "#9A93B5"
BG = "#FFFFFF"
# Positron без подписей и границ: суша и вода, без линий
WATER = "#B8C5C9"
LAND_POS = "#F7F7F4"

# три типа лендинга собраны из пяти типов модели
GROUP = {"match": "similar", "match_partial": "similar", "match_expanded": "similar",
         "merge": "macro", "new_market": "new"}
COLOR = {"similar": "#A48DF5", "macro": BRAND, "new": "#FF6B3D"}
LABEL = {"similar": "Совпадает или похож на официальную",
         "macro": "Макроагломерация",
         "new": "Новый рыночный кластер"}

def load(name):
    # GDAL не открывает пути с «!», поэтому читаем через json
    return gpd.GeoDataFrame.from_features(json.loads((HERE / name).read_text())["features"], crs=4326)


mo = load("clusters.geojson")
off = load("official.geojson")
land = gpd.GeoDataFrame.from_features(
    json.loads((HERE / "img" / "ne_land.geojson").read_text())["features"], crs=4326
)
mo["group"] = [GROUP.get(t) if s >= 3 else None for t, s in zip(mo["type"], mo["cluster_size"])]
regions = mo.dissolve("region_name")


def local_crs(geom):
    c = geom.centroid
    return f"+proj=aeqd +lat_0={c.y:.3f} +lon_0={c.x:.3f} +units=m"


def halo(txt, w=4):
    txt.set_path_effects([patheffects.withStroke(linewidth=w, foreground="white")])


def example(cluster_id, group, labels, fname, pad=0.35, official_names=None, show_other=False, bbox_from=None):
    focus = mo[mo["cluster"] == cluster_id]
    extent = focus
    if bbox_from:
        mask = False
        for part in bbox_from:
            mask = mask | focus["municipal_district_name"].str.contains(part, regex=False)
        extent = focus[mask]
    crs = local_crs(extent.to_crs(4326).union_all())
    f = focus.to_crs(crs)
    xmin, ymin, xmax, ymax = extent.to_crs(crs).total_bounds
    w, h = xmax - xmin, ymax - ymin
    side = max(w, h * 4 / 3) * (1 + 2 * pad)
    # кадр смещён вниз, чтобы легенда в левом нижнем углу не перекрывала кластер
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2 - side * 0.05
    box = (cx - side / 2, cy - side * 3 / 8, cx + side / 2, cy + side * 3 / 8)

    m = mo.to_crs(crs).cx[box[0]:box[2], box[1]:box[3]]
    r = regions.to_crs(crs).cx[box[0]:box[2], box[1]:box[3]]
    o = off.to_crs(crs).cx[box[0]:box[2], box[1]:box[3]]
    if official_names is not None:
        o = o[o["agg"].isin(official_names)]

    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
    fig.patch.set_facecolor(BG)
    # фон = цвет суши, чтобы МО без данных не выглядели дырами
    ax.set_facecolor(LAND)
    m.plot(ax=ax, color=LAND, edgecolor=MO_EDGE, linewidth=0.8)
    if show_other:
        other = m[m["group"].notna() & (m["cluster"] != cluster_id)]
        for g, part in other.groupby("group"):
            part.plot(ax=ax, color=COLOR[g], alpha=0.28, edgecolor=MO_EDGE, linewidth=0.8)
    f.plot(ax=ax, color=COLOR[group], edgecolor=MO_EDGE, linewidth=1.0)
    f.dissolve().boundary.plot(ax=ax, color=COLOR[group], linewidth=2.5)
    r.boundary.plot(ax=ax, color=REGION_EDGE, linewidth=1.4, linestyle=(0, (1, 2)))
    if len(o):
        o.boundary.plot(ax=ax, color=INK, linewidth=2.6, linestyle=(0, (5, 3)))

    for name, (match, dx, dy) in labels.items():
        row = m[m["municipal_district_name"].str.contains(match, regex=False)]
        if row.empty:
            continue
        p = row.geometry.iloc[0].representative_point()
        ax.plot(p.x, p.y, "o", ms=12, color="white", mec=INK, mew=2.5, zorder=5)
        t = ax.annotate(name, (p.x, p.y), xytext=(dx, dy), textcoords="offset points",
                        ha="right" if dx < 0 else "left", va="center",
                        fontsize=32, fontweight="bold", color=INK, zorder=6)
        halo(t, 7)

    handles = [Patch(facecolor=COLOR[group], edgecolor="none", label=LABEL[group]),
               Line2D([], [], color=INK, lw=2.6, ls=(0, (5, 3)), label="Официальная агломерация"),
               Line2D([], [], color=REGION_EDGE, lw=1.6, ls=(0, (1, 2)), label="Граница региона")]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=22, frameon=True, borderpad=0.9,
                    handlelength=2.6, labelspacing=0.7)
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_alpha(0.92)

    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[1], box[3])
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / fname, facecolor=LAND)
    plt.close(fig)
    print("→", OUT / fname)


RUSSIA = "+proj=aea +lat_1=50 +lat_2=70 +lat_0=56 +lon_0=100 +units=m"


def russia_map(fname, draw):
    m = mo.to_crs(RUSSIA)
    fig, ax = plt.subplots(figsize=(20, 10), dpi=120)
    fig.patch.set_facecolor(WATER)
    ax.set_facecolor(WATER)
    land.cx[10:180, 35:82].to_crs(RUSSIA).plot(ax=ax, color=LAND_POS, edgecolor="none")
    draw(ax)
    xmin, ymin, xmax, ymax = m[m.geometry.centroid.x < 3.2e6].total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / fname, facecolor=WATER, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("→", OUT / fname)


def draw_official(ax):
    off.to_crs(RUSSIA).plot(ax=ax, color=INK, alpha=0.82, edgecolor="none")


def draw_clusters(ax):
    c = load("cluster_outlines.geojson")
    c[c["cluster_size"] >= 3].to_crs(RUSSIA).plot(ax=ax, color=BRAND, alpha=0.85, edgecolor="none")


if __name__ == "__main__":
    example(490, "similar", {
        "Киров": ("город Киров", -18, 0),
        "Кирово-Чепецк": ("город Кирово-Чепецк", 18, -22),
        "Слободской": ("город Слободской", 18, 14),
    }, "type_similar_kirov.png", pad=0.12, official_names=["Кировская"])
    example(114, "macro", {
        "Казань": ("город Казань", 18, 0),
        "Чебоксары": ("город Чебоксары", 18, 0),
        "Ульяновск": ("город Ульяновск", 18, 0),
    }, "type_macro_kazan.png", pad=0.06,
        official_names=["Казанская", "Чебоксарская", "Ульяновско-Димитровградская"])
    example(113, "new", {
        "Йошкар-Ола": ("город Йошкар-Ола", 18, 0),
        "Яранск": ("Яранский", 18, 0),
        "Советск": ("Советский муниципальный район", 18, 0),
    }, "type_new_yoshkar.png", pad=0.1)
    example(379, "new", {
        "Архангельск": ("город Архангельск", 18, 0),
        "Северодвинск": ("Северодвинск", 18, 0),
        "Новодвинск": ("город Новодвинск", 18, 12),
    }, "type_new_arkhangelsk.png", pad=0.35,
        bbox_from=["город Архангельск", "Северодвинск", "город Новодвинск"])
    example(745, "new", {
        "Смоленск": ("город Смоленск", 18, 0),
        "Сафоново": ("Сафоновский", 18, 0),
        "Ярцево": ("Ярцевский", 18, 0),
    }, "type_new_smolensk.png", pad=0.12)
    example(404, "new", {
        "Муром": ("округ Муром", 18, 0),
        "Выкса": ("город Выкса", 18, 0),
        "Кулебаки": ("город Кулебаки", 18, 0),
    }, "type_new_murom.png", pad=0.18)
    example(192, "new", {
        "Бийск": ("город Бийск", 18, 0),
        "Белокуриха": ("город Белокуриха", 18, 0),
        "Алтайский": ("Алтайский муниципальный район", 18, 0),
    }, "type_new_biysk.png", pad=0.12)
    russia_map("official.png", draw_official)
    russia_map("clusters.png", draw_clusters)
