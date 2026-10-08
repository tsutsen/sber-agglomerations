"""Экспорт финальных кластеров в GeoJSON для лэндинга."""
from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd

PAL = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628",
       "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]

g = gpd.read_file("data/mo_consumption_population.gpkg")
cl = pd.read_csv("out/clusters_all.csv")
final = cl[cl.window == cl.window.max()].drop_duplicates("mo").rename(columns={"mo": "territory_id"})
m = g.merge(final, on="territory_id", how="inner")
rep = pd.read_csv("out/cluster_vs_official.csv")
agg_by = dict(zip(rep.cluster, rep.dominant_agg.fillna("")))

m["type"] = m.cluster.map(rep.set_index("cluster").type.fillna("small"))
m["n_mo"] = m.cluster.map(rep.set_index("cluster").n_mo)
m["dominant"] = m.cluster.map(agg_by)
m["color"] = m.type.where(m.type == "small", "#bbbbbb")
for cid, grp in m[m.type != "small"].groupby("cluster"):
    m.loc[grp.index, "color"] = PAL[int(cid) % len(PAL)]

props = m[["territory_id", "municipal_district_name", "region_name", "cluster",
           "type", "n_mo", "dominant", "color"]].rename(
    columns={"n_mo": "cluster_size"})
geom = m.geometry.simplify(0.005, preserve_topology=True)
out = gpd.GeoDataFrame(props, geometry=geom).to_json()
with open("data/clusters.geojson", "w") as f:
    f.write(out)
print("written", len(m), "features -> data/clusters.geojson")
