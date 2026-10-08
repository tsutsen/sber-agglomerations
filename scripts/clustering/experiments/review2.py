"""Прогонки по второму внешнему ревью: абляции level vs structure, run-to-run ARI,
к-чувствительность, holdout по ФО, внутригородская чувствительность «новых» кластеров."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..features import FEAT_COLS, load_features
from ..network.graphs import candidate_pairs, louvain, window_edges
from sklearn.metrics import adjusted_rand_score

FO_MAP = {}
ALIASES = {
    "Республика Татарстан": "Татарстан", "Удмуртская Республика": "Республика Удмуртия",
    "Чувашская Республика": "Чувашия", "Нижегородская область": "Нижнегородская область",
    "Ханты-Мансийский автономный округ — Югра": "ХМАО",
    "Ямало-Ненецкий автономный округ": "ЯНАО",
    "Еврейская автономная область": "Еврейская АО", "Чукотский автономный округ": "Чукотский АО",
}
for name, regs in [
    ("ЦФО", ["Белгородская область", "Брянская область", "Владимирская область", "Воронежская область",
             "Ивановская область", "Калужская область", "Костромская область", "Курская область",
             "Липецкая область", "Московская область", "Москва", "Орловская область",
             "Рязанская область", "Смоленская область", "Тамбовская область", "Тверская область",
             "Тульская область", "Ярославская область", "Калининградская область"]),
    ("СЗФО", ["Санкт-Петербург", "Ленинградская область", "Архангельская область", "Вологодская область",
              "Мурманская область", "Новгородская область", "Псковская область",
              "Республика Карелия", "Республика Коми"]),
    ("ПФО", ["Кировская область", "Нижнегородская область", "Оренбургская область", "Пензенская область",
             "Пермский край", "Республика Башкортостан", "Республика Марий Эл",
             "Республика Мордовия", "Республика Удмуртия", "Самарская область", "Саратовская область",
             "Татарстан", "Ульяновская область", "Чувашия"]),
    ("УФО", ["Курганская область", "Свердловская область", "Тюменская область", "Челябинская область",
             "ХМАО", "ЯНАО", "Республика Бурятия", "Ненецкий автономный округ",
             "Республика Саха (Якутия)"]),
    ("СФО", ["Алтайский край", "Забайкальский край", "Иркутская область", "Кемеровская область",
             "Красноярский край", "Новосибирская область", "Омская область", "Томская область",
             "Республика Тыва", "Республика Хакасия", "Республика Алтай"]),
    ("СКФО", ["Республика Дагестан", "Республика Ингушетия", "Кабардино-Балкарская Республика",
             "Карачаево-Черкесская Республика", "Республика Северная Осетия — Алания",
             "Чеченская Республика"]),
    ("ЮФО", ["Астраханская область", "Волгоградская область", "Ростовская область", "Краснодарский край",
             "Республика Адыгея", "Республика Калмыкия", "Республика Крым", "Севастополь",
             "Ставропольский край"]),
    ("ДВФО", ["Амурская область", "Еврейская АО", "Камчатский край", "Магаданская область",
              "Приморский край", "Сахалинская область", "Хабаровский край", "Чукотский АО"]),
]:
    for r in regs:
        FO_MAP[r] = name
# уточнения по официальным границам ФО
FO_MAP["Республика Саха (Якутия)"] = "ДВФО"
FO_MAP["Ненецкий автономный округ"] = "СЗФО"


def main() -> None:
    feats, months = load_features()
    ids = np.sort(feats.mo.unique())
    pos = {int(t): i for i, t in enumerate(ids)}

    import geopandas as gpd
    g = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                    usecols=["territory_id", "region_name"]).drop_duplicates("territory_id")
    reg = dict(zip(g.territory_id, g.region_name))
    mo_fo = np.array([FO_MAP.get(ALIASES.get(reg[int(t)], reg[int(t)]), "?") for t in ids])
    print("ФО покрытие:", {k: int((mo_fo == k).sum()) for k in np.unique(mo_fo)})

    off = pd.read_csv("data/experiments/inputs/derived/official_agglomerations_mapped.csv")
    off = off[off.territory_id.notna()]
    agg_by_id = dict(zip(off.territory_id, off["Агломерация"]))
    uniq = {v: i for i, v in enumerate(sorted({agg_by_id[int(t)] for t in agg_by_id if int(t) in pos}))}
    y = np.full(len(ids), -1, dtype=int)
    for t, a in agg_by_id.items():
        if int(t) in pos:
            y[pos[int(t)]] = uniq[a]
    off_mask = y >= 0
    n_off_mo = int(off_mask.sum())
    print(f"МО официального перечня в выборке: {n_off_mo}")
    fo_off = mo_fo[off_mask]

    t_end = 23

    def purity_of(members: np.ndarray) -> float:
        tot, hit = 0, 0
        for cid in np.unique(members):
            mem = (members == cid)
            counts: dict[str, int] = {}
            for t in ids[mem][off_mask[mem]]:
                a = agg_by_id.get(int(t))
                if a:
                    counts[a] = counts.get(a, 0) + 1
            if counts:
                dom = max(counts.values())
                tot += sum(counts.values())
                hit += dom
        return hit / tot if tot else 0.0

    def ari_on(members: np.ndarray, m: np.ndarray) -> float:
        return float(adjusted_rand_score(y[m], members[m]))

    rows = []

    # --- 1. production + run-to-run
    pairs, _ = candidate_pairs("data/sources", ids)
    edges = window_edges(feats, t_end, pairs, ids, k=8)
    runs = [louveau() for _ in range(6)] if False else [louvain(len(ids), edges)[0] for _ in range(6)]
    pw = [adjusted_rand_score(runs[i], runs[j]) for i in range(6) for j in range(i + 1, 6)]
    m0 = runs[0]
    a = ari_on(m0, off_mask)
    n3 = sum(1 for c in np.unique(m0) if (m0 == c).sum() >= 3)
    rows.append({"run": "production k=8", "edges": len(edges), "ari": round(a, 3),
                 "purity": round(purity_of(m0), 3), "clusters": len(np.unique(m0)), "n3": n3})
    print(f"production: ARI={a:.3f} purity={purity_of(m0):.3f} clusters={len(np.unique(m0))} "
          f"run-to-run mean pairwise ARI={np.mean(pw):.4f} (min {np.min(pw):.4f})")

    # --- 2. абляции
    w = feats[(feats.t >= t_end - 5) & (feats.t <= t_end)]
    prof = w.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0).to_numpy(float)
    shares, level = prof[:, :5], prof[:, 5:6]
    Zs = (shares - shares.mean(0)) / shares.std(0, ddof=0).clip(min=1e-9)
    Zl = (level - level.mean()) / level.std(ddof=0)
    Z6 = np.column_stack([Zs, Zl])

    def topk_edges(vec: np.ndarray, k: int = 8) -> list:
        norm = np.linalg.norm(vec, axis=1).clip(min=1e-9)
        a_, b_ = pairs[:, 0], pairs[:, 1]
        cos = (vec[a_] * vec[b_]).sum(1) / (norm[a_] * norm[b_])
        cos = np.maximum(cos, 0.0)  # igraph не принимает отрицательные веса
        df = pd.DataFrame({"i": a_, "j": b_, "w": cos})
        df2 = df.copy()
        df2["i2"], df2["j2"] = b_, a_
        both = pd.concat([df[["i", "j", "w"]], df2[["i2", "j2", "w"]]], ignore_index=True)
        both = both.sort_values("w", ascending=False).groupby("i").head(k)
        bi, bj = both.i.to_numpy(), both.j.to_numpy()
        lo, hi = np.minimum(bi, bj), np.maximum(bi, bj)
        uniq = dict(zip(zip(lo, hi), both.w.to_numpy(), strict=True))
        return [(int(i), int(j), float(v)) for (i, j), v in uniq.items()]

    for name, vec in [("ablation: log-level only (1d)", level),
                      ("ablation: 5 shares raw (no level)", shares),
                      ("ablation: z-shares (5d)", Zs),
                      ("control: raw 6d (production)", prof),
                      ("control: z6", Z6)]:
        ed = topk_edges(vec)
        mem = louvain(len(ids), ed)[0]
        a = ari_on(mem, off_mask)
        n3 = sum(1 for c in np.unique(mem) if (mem == c).sum() >= 3)
        rows.append({"run": name, "edges": len(ed), "ari": round(a, 3),
                     "purity": round(purity_of(mem), 3), "clusters": len(np.unique(mem)), "n3": n3})
        print(f"{name}: edges={len(ed)} ARI={a:.3f} purity={purity_of(mem):.3f} "
              f"clusters={len(np.unique(mem))}")

    # --- 3. k-чувствительность
    for k in (5, 12):
        ed = window_edges(feats, t_end, pairs, ids, k=k)
        mem = louvain(len(ids), ed)[0]
        a = ari_on(mem, off_mask)
        rows.append({"run": f"production k={k}", "edges": len(ed), "ari": round(a, 3),
                     "purity": round(purity_of(mem), 3), "clusters": len(np.unique(mem))})
        print(f"production k={k}: edges={len(ed)} ARI={a:.3f}")

    # --- 4. holdout по ФО (детерминированный сплит по числу МО перечня)
    cnts = {fo: int((fo_off == fo).sum()) for fo in np.unique(fo_off)}
    sorted_fos = sorted(cnts, key=lambda f: (-cnts[f], f))
    tune_fos, val_fos = set(sorted_fos[::2]), set(sorted_fos[1::2])
    print(f"tune ФО {sorted(tune_fos)} ({sum(cnts[f] for f in tune_fos)} МО); "
          f"val ФО {sorted(val_fos)} ({sum(cnts[f] for f in val_fos)} МО)")
    for cutoff in (40.0, 50.0, 60.0):
        pc, _ = candidate_pairs("data/sources", ids, cutoff=cutoff)
        ec = window_edges(feats, t_end, pc, ids, k=8)
        mem = louvain(len(ids), ec)[0]
        mt = off_mask & np.isin(mo_fo, list(tune_fos))
        mv = off_mask & np.isin(mo_fo, list(val_fos))
        at, av = ari_on(mem, mt), ari_on(mem, mv)
        rows.append({"run": f"holdout cutoff={cutoff:.0f}", "edges": len(ec),
                     "ari_tune": round(at, 3), "ari_validate": round(av, 3)})
        print(f"cutoff={cutoff:.0f}: ARI_tune={at:.3f} ARI_validate={av:.3f}")

    pd.DataFrame(rows).to_csv("out/review2_ablations.csv", index=False)

    # --- 5. «новые» кластеры: внутригородская чувствительность
    cv = pd.read_csv("out/cluster_vs_official.csv")
    new_clusters = set(cv[cv["n_official_agg"] == 0]["cluster"])
    ca = pd.read_csv("out/clusters_all.csv")
    final = ca[ca["window"] == ca["window"].max()]
    nc = final[final["cluster"].isin(new_clusters)]
    msk_ids = set(int(t) for t in ids if reg[int(t)] == "Москва")
    spb_ids = set(int(t) for t in ids if reg[int(t)] == "Санкт-Петербург")
    intra = nc[nc["mo"].isin(msk_ids | spb_ids)]
    print(f"новых кластеров (не в перечне): {len(new_clusters)}; МО в них: {len(nc)}; "
          f"из них внутригородские МСК/СПб: {len(intra)} "
          f"(МСК {intra['mo'].isin(msk_ids).sum()}, СПб {intra['mo'].isin(spb_ids).sum()})")
    cand = []
    for cid, grp in nc.groupby("cluster"):
        if len(grp) < 3:
            continue
        cand.append({
            "cluster": int(cid), "n_mo": int(len(grp)),
            "has_intracity_msk_spb": bool(grp["mo"].isin(msk_ids | spb_ids).any()),
            "sample_regions": sorted({reg[int(t)] for t in grp["mo"].unique()})[:4],
        })
    cand.sort(key=lambda x: -x["n_mo"])
    with open("out/review2_new_clusters.json", "w", encoding="utf-8") as f:
        json.dump(cand, f, ensure_ascii=False, indent=1)
    print("новых кластеров >=3 МО:", len(cand),
          "| без внутригородских МСК/СПб:", sum(1 for c in cand if not c["has_intracity_msk_spb"]))


if __name__ == "__main__":
    main()
