"""EXP 2: компактность кластеров Polsby–Popper (Lattimer 2022).

PP = 4*pi*A/P^2  (1 = круг, 0 = бесконечно вытянутая фигура).
Статья: PP — стандартный географический индекс качества административных
регионов; компактность — одно из пяти свойств регионализации (SKATER).

Считаем PP для (а) 872 кластеров производства (полигоны из
data/cluster_outlines.geojson — объединение контуров МО) и (б) регионов
типологии 14 осей (объединение контуров МО того же типа — типология без
ограничения связности, PP показывает, насколько её регионы географически
разлиты).

Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.paper_exp2_pp.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyproj
from shapely.geometry import shape
from shapely.ops import transform, unary_union

import geopandas as gpd

from .papers_common import load_ctx, typ14


def pp(geom) -> float:
    """Polsby–Popper в UTM-зоне центроиды (метры), иначе искажение в 4326."""
    lon = geom.centroid.x
    zone = int((lon + 180) / 6) % 60 + 1
    crs = f"EPSG:326{zone:02d}"
    t = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    g = transform(t.transform, geom)
    a, p = g.area, g.length
    return 4 * np.pi * a / (p * p) if p > 0 else 0.0


def main() -> None:
    feats, ids, pairs, _ = load_ctx()
    g = gpd.read_file("data/results/geojson/mo.geojson", columns=["territory_id"])
    g.geometry = g.geometry.make_valid()  # в mo.geojson есть слабые топонические пересечения
    g = g.set_index(g.territory_id.to_numpy())
    g = g.loc[ids]  # порядок = ids
    geo_by_id = g.geometry.to_dict()

    # ---- production: готовые полигоны ----
    gj = json.load(open("data/experiments/inputs/derived/cluster_outlines.geojson"))
    rows = []
    for f in gj["features"]:
        if f.get("geometry") is None:
            continue
        p = f["properties"]
        rows.append({"partition": "production", "id": int(p["cluster"]),
                     "name": p.get("name", str(p["cluster"])),
                     "size": int(p.get("cluster_size", 0)),
                     "PP": round(pp(shape(f["geometry"])), 4)})
    prod = pd.DataFrame(rows)

    # ---- typology: объединение контуров МО по типу ----
    X, cols, mt, q = typ14(feats, ids, pairs)
    by_type: dict[int, list] = {}
    for i, tid in enumerate(ids):
        by_type.setdefault(int(mt[i]), []).append(geo_by_id[tid])
    rows = []
    for t, geoms in by_type.items():
        rows.append({"partition": "typology", "id": int(t),
                     "name": f"type_{t}", "size": len(geoms),
                     "PP": round(pp(unary_union(geoms)), 4)})
    typ = pd.DataFrame(rows)

    out = pd.concat([prod, typ], ignore_index=True)
    out.to_csv("out/paper_exp2_pp.csv", index=False)

    # ---- сводка ----
    def summ(d: pd.DataFrame, label: str) -> dict:
        s = d.PP
        return {"part": label, "n": len(d), "PP_med": round(float(s.median()), 3),
                "PP_p10": round(float(s.quantile(0.1)), 3),
                "PP_p90": round(float(s.quantile(0.9)), 3),
                "PP_min": round(float(s.min()), 3), "PP_gt0.5": int((s > 0.5).sum())}

    # null: случайные объединения МО того же размера (географически разлито)
    rng = np.random.default_rng(42)
    all_mo = list(geo_by_id.values())
    rand_pp = []
    for s in prod.loc[prod["size"] >= 3, "size"]:
        for _ in range(20):
            sample = rng.choice(all_mo, size=int(s), replace=False)
            rand_pp.append(pp(unary_union(list(sample))))
    def summ_vals(label: str, vals: list[float]) -> dict:
        v = np.asarray(vals)
        return {"part": label, "n": len(v), "PP_med": round(float(np.median(v)), 3),
                "PP_p10": round(float(np.quantile(v, 0.1)), 3),
                "PP_p90": round(float(np.quantile(v, 0.9)), 3),
                "PP_min": round(float(v.min()), 3),
                "PP_gt0.5": int((v > 0.5).sum())}

    # null для типологии: случайные объединения тех же (крупных) размеров
    rand_pp_t = []
    for s in typ["size"]:
        for _ in range(10):
            sample = rng.choice(all_mo, size=int(s), replace=False)
            rand_pp_t.append(pp(unary_union(list(sample))))

    summary = pd.DataFrame([summ(prod, "production (кластеры >=1)"),
                            summ(prod[prod["size"] >= 3], "production (кластеры >=3)"),
                            summ(typ, "typology14 (регионы)"),
                            summ_vals("null: случайные (размеры кластеров)", rand_pp),
                            summ_vals("null: случайные (размеры типологии)", rand_pp_t)])
    summary.to_csv("out/paper_exp2_pp_summary.csv", index=False)

    print("\n=== сводка ===")
    print(summary.to_string(index=False))
    print("\n=== production, самые вытянутые (PP < 0.2, size>=3) ===")
    print(prod[(prod.PP < 0.2) & (prod["size"] >= 3)]
          .sort_values("PP").head(12).to_string(index=False))
    print("\n=== typology: PP по типам ===")
    print(typ.sort_values("PP").to_string(index=False))


if __name__ == "__main__":
    main()
