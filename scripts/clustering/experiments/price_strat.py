"""Стратифицированная проверка: эффект «ближе цены у рёбер» после маппинга
межрегиональных пар по |Δ log-уровня| (отделяет level от structure)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..features import FEAT_COLS, load_features
from ..network.graphs import candidate_pairs

t_end = 23


def main() -> None:
    feats, _ = load_features()
    ids = np.sort(feats.mo.unique())
    w = feats[(feats.t >= t_end - 5) & (feats.t <= t_end)]
    prof = w.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0).to_numpy(float)
    level = prof[:, 5]
    st = pd.read_csv("data/experiments/inputs/derived/mo_static_features.csv") if False else None
    import geopandas as gpd
    g = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                    usecols=["territory_id", "region_name"]).drop_duplicates("territory_id")
    reg = dict(zip(g.territory_id, g.region_name))
    basket = pd.read_csv("data/sources/basket_size_per_mo.csv")
    MON = {n: i + 1 for i, n in enumerate(
        ["январь", "февраль", "март", "апрель", "май", "июнь",
         "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"])}
    basket["t"] = (basket.year - 2023) * 12 + basket.month.map(MON) - 1
    coef24 = basket.pivot_table(index="region_name", columns="t", values="value").iloc[:, 12:24].mean(axis=1)
    coef = {int(t): float(coef24.get(reg[int(t)], np.nan)) for t in ids}

    edges = set()
    for line in open("out/edges_2024-12.csv"):
        if line.startswith("i,"):
            continue
        i, j, _ = line.strip().split(",")
        edges.add((int(i), int(j)))

    pairs, _ = candidate_pairs("data/sources", ids)
    rows = []
    for a, b in pairs:
        ra, rb = reg.get(int(ids[a])), reg.get(int(ids[b]))
        if not ra or ra == rb:
            continue
        ca, cb = coef.get(int(ids[a]), np.nan), coef.get(int(ids[b]), np.nan)
        if np.isnan(ca) or np.isnan(cb):
            continue
        gap = abs(ca - cb) / min(ca, cb)
        dlevel = abs(level[a] - level[b])
        is_edge = (a, b) in edges or (b, a) in edges
        rows.append({"dlevel": dlevel, "gap": gap, "edge": is_edge})
    df = pd.DataFrame(rows)
    print("межрегиональных пар с coef:", len(df), "из них рёбер:", int(df.edge.sum()))
    q1, q2 = df.dlevel.quantile([0.333, 0.667]).tolist()
    df["bin"] = pd.cut(df.dlevel, [0, q1, q2, np.inf], labels=["низкий ΔL", "средний", "высокий"])
    out = []
    for bname, grp in df.groupby("bin", observed=True):
        e, n = grp[grp.edge], grp[~grp.edge]
        out.append({
            "dlevel_bin": str(bname), "n_edge": len(e), "n_nedge": len(n),
            "edge_gap_lt005": round(float((e.gap < 0.05).mean()), 3) if len(e) else None,
            "nedge_gap_lt005": round(float((n.gap < 0.05).mean()), 3),
            "edge_gap_lt010": round(float((e.gap < 0.10).mean()), 3) if len(e) else None,
            "nedge_gap_lt010": round(float((n.gap < 0.10).mean()), 3),
        })
        print(out[-1])
    json.dump(out, open("out/price_strat_dlevel.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("→ out/price_strat_dlevel.json")


if __name__ == "__main__":
    main()
