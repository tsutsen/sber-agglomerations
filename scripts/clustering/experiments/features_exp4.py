"""G-варианты: цена стандартной корзины (региональный ценовой уровень)
+ национальная динамика (два CSV СберИндекса, оба — по России).

Источники:
- data/sources/basket_size_per_mo.csv (индекс корзины по регионам)
  — коэффициент цены стандартной корзины, 85 регионов, 2023–2025 (мес.).
  Чукотка 1.67 ... Мордовия 0.86 — региональный ценовой уровень.
- data/experiments/inputs/raw/sber_national_price_index_by_category.csv
  — месячные индексы (база 100) по 5 типам + «Всего», Россия.
- data/experiments/inputs/raw/sber_national_category_growth.csv
  — недельные % г/г по 46 категориям, Россия.

Варианты (правило рёбер как в производстве: <=50 км + топ-8):
- G2 real:   log_total_real = log(total / basket_coef(t)) — дефлирование
             номинала региональным ценовым уровнем (реальное потребление).
- G4 nation: кафе/индекс_общпит, прод/индекс_прод, прочее и total/индекс_всего.
- G1 grow:   11 признаков = 6 dyn + 5 «избыточный рост» (средний YoY-рост МО
             в 2024 по категории − средний YoY России), все z-scored.
- G3 typology: типология на 14 осях (12 + log_basket + growth_basket).

Порог выбора — тот же, что в features_exp (ARI +0.05 ИЛИ purity +0.03).
Вывод: out/features_exp4.csv, out/types_profile_g3.csv

Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.features_exp4.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features_exp import (KNN, STATIC, T_END, ari_all, louvain,
                                  run_typology, window_edges_cols, zscore)
from ..features import CATS, FEAT_COLS, load_features
from ..network.graphs import candidate_pairs
from ..tune import purity

BASKET = "data/sources/basket_size_per_mo.csv"
IDX_RU = "data/experiments/inputs/raw/sber_national_price_index_by_category.csv"
YOY_RU = "data/experiments/inputs/raw/sber_national_category_growth.csv"
RU_YOY_MAP = {  # наши категории -> категория(и) файла с недельными % г/г
    "cafe": ["Общественное питание"],
    "food": ["Продовольственные товары"],
    "health": ["Лекарства и медицинские товары"],
    "marketplace": ["Маркетплейсы"],
    "transport": ["Локальный транспорт", "Топливо"],
}
MONTHS_RU = {n: i + 1 for i, n in enumerate(
    ["январь", "февраль", "март", "апрель", "май", "июнь",
     "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"])}


def basket_coef_by_mo() -> np.ndarray:
    """(n_mo, 24) коэффициент корзины, строки в порядке ids, NaN если нет."""
    b = pd.read_csv(BASKET)
    b["month"] = b.month.map(MONTHS_RU)
    b["t"] = (b.year - 2023) * 12 + (b.month - 1)
    b = b[b.t.isin(range(24))]
    pop = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                      usecols=["territory_id", "region_name"])
    j = b.merge(pop, on="region_name", how="inner")
    print(f"  basket: МО покрыто {j.territory_id.nunique()} "
          f"из {pop.territory_id.nunique()}")
    return j.pivot(index="territory_id", columns="t", values="value")


def smoothed_cats() -> pd.DataFrame:
    """[mo, t, s_{cat}, s_total] — сглаженные категории, как в features.py."""
    m = pd.read_csv("data/sources/derived/mo_consumption_monthly_long.csv",
                    usecols=["territory_id", "date", "cat", "consumption_rub"])
    m = m[m.cat.isin(CATS)]
    p = m.pivot_table(index=["territory_id", "date"], columns="cat",
                      values="consumption_rub", aggfunc="mean").reset_index()
    months = sorted(p.date.unique())
    p["t"] = p.date.map({d: i for i, d in enumerate(months)})
    for c in CATS:
        p[c] = p[c].fillna(0.0)
    p = p.sort_values(["territory_id", "t"])
    for c in CATS:
        p[f"s_{c}"] = p.groupby("territory_id")[c].transform(
            lambda s: s.rolling(3, min_periods=1).mean())
    p["s_total"] = p[[f"s_{c}" for c in CATS]].sum(axis=1)
    return p.rename(columns={"territory_id": "mo"})


def run_g(name, edges_fn, ids, agg_by_id, n_runs=3) -> list[dict]:
    """n_runs запусков Louvain на фиксированных рёбрах -> разбег метрик."""
    rows = []
    for r in range(n_runs):
        members, q = louvain(len(ids), edges_fn())
        rows.append({"variant": name, "edges": None,
                     "clusters": int(members.max() + 1),
                     "q": round(float(q), 3),
                     "purity": round(purity(members, ids, agg_by_id), 3),
                     "ari": round(ari_all(members, ids, agg_by_id), 3)})
    rows[0]["edges"] = len(edges_fn())
    for r in rows[1:]:
        r["edges"] = rows[0]["edges"]
    return rows


def run_typology14(feats, static, ids, pairs, agg_by_id, stat_cols):
    """Копия run_typology с произвольным набором статических осей."""
    dyn = feats.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0)
    stat = static[stat_cols].reindex(ids).fillna(0.0)
    X = zscore(np.hstack([dyn.to_numpy(dtype=float), stat.to_numpy(dtype=float)]))
    cols = FEAT_COLS + stat_cols
    norm = np.linalg.norm(X, axis=1).clip(min=1e-9)
    sim = (X / norm[:, None]) @ (X / norm[:, None]).T
    np.fill_diagonal(sim, -1.0)
    sim = np.where(sim > 0, sim, -1.0)
    idx = np.argsort(-sim, axis=1)[:, :KNN]
    iu = np.repeat(np.arange(len(ids)), KNN)
    ju = idx.ravel()
    w = sim[iu, ju]
    ok = w > 0
    lo, hi = np.minimum(iu[ok], ju[ok]), np.maximum(iu[ok], ju[ok])
    uniq = dict(zip(zip(lo, hi), w[ok], strict=True))
    edges = [(int(i), int(j), float(w)) for (i, j), w in uniq.items()]
    members, q = louvain(len(ids), edges)

    nb = {}
    for a, b in zip(*pairs.T):
        nb.setdefault(int(a), set()).add(int(b))
        nb.setdefault(int(b), set()).add(int(a))

    def coherence(m: np.ndarray) -> float:
        return sum(any(m[j] == m[i] for j in nb.get(i, ())) for i in range(len(m))) / len(m)

    coh = coherence(members)
    rng = np.random.default_rng(42)
    shuffled = [coherence(rng.permutation(members)) for _ in range(1000)]
    res = {"variant": "G3_typology14", "edges": len(edges),
           "clusters": int(members.max() + 1),
           "clusters_5plus": int(sum((members == c).sum() >= 5 for c in np.unique(members))),
           "q": round(float(q), 3), "spatial_coh": round(coh, 3),
           "spatial_coh_null_max": round(max(shuffled), 3),
           "p_value": round((sum(s >= coh for s in shuffled) + 1) / 1001, 4),
           "ari": round(ari_all(members, ids, agg_by_id), 3)}
    rows = []
    for cid in np.unique(members):
        mem = members == cid
        if mem.sum() < 5:
            continue
        z = X[mem].mean(0)
        top = np.argsort(-np.abs(z))[:3]
        rows.append({"cluster": int(cid), "n_mo": int(mem.sum()),
                     **{c: round(float(z[j]), 2) for j, c in enumerate(cols)},
                     "top3": " > ".join(cols[j] for j in top)})
    profile = pd.DataFrame(rows).sort_values("n_mo", ascending=False)
    return res, profile, members


def main() -> None:
    feats, months = load_features()
    ids = np.sort(feats.mo.unique())
    pairs, _ = candidate_pairs("data/sources", ids)
    off = pd.read_csv("data/experiments/inputs/derived/official_agglomerations_mapped.csv")
    off = off[off.territory_id.notna()]
    agg_by_id = dict(zip(off.territory_id, off["Агломерация"]))
    pos = {mo: i for i, mo in enumerate(ids)}
    rows: list[dict] = []

    # ---- G2: реальное потребление (дефлирование ценовым уровнем) ----
    bmat = basket_coef_by_mo()  # index territory_id, columns t
    bc = np.full((len(ids), 24), np.nan)
    for i, mo in enumerate(ids):
        if mo in bmat.index:
            bc[i] = bmat.loc[mo].values
    with np.errstate(invalid="ignore"):
        bc = np.where(np.isnan(bc), np.nanmedian(bc, axis=1, keepdims=True), bc)
    real = np.log1p(feats.total.to_numpy() / bc[feats.mo.map(pos).to_numpy(), feats.t.to_numpy()])
    feats2 = feats.copy()
    feats2["log_total_real"] = real
    cols_g2 = [f"share_{c}" for c in CATS] + ["log_total_real"]
    rows += run_g("G2 real (total/корзина)",
                  lambda: window_edges_cols(feats2, T_END, pairs, ids, cols_g2),
                  ids, agg_by_id)

    # ---- G4: национальная дефляция индексами РФ ----
    rb = pd.read_csv(IDX_RU, sep=";", encoding="utf-8-sig")
    rb["t"] = rb.period.map(lambda d: (int(d[:4]) - 2023) * 12 + int(d[5:7]) - 1)
    base = rb[rb.t == 0].set_index("type").value
    rb["norm"] = rb.value / rb.type.map(base) * 100
    norm = rb[rb.t.isin(range(24))].set_index(["type", "t"]).norm.to_dict()
    nfn = {"cafe": "Общественное питание", "food": "Продовольственные товары"}
    s = smoothed_cats()
    sp = s[s.t.isin(range(24))].copy()
    nt = np.array([norm.get(("Всего", t), 100.0) for t in sp.t])
    for c in CATS:
        n = np.array([norm.get((nfn[c], t), 100.0) for t in sp.t]) if c in nfn else nt
        sp[f"d_{c}"] = sp[f"s_{c}"] / n
    sp["d_total"] = sp.s_total / nt
    f4 = feats[["mo", "t"]].merge(
        sp[["mo", "t", "d_total"] + [f"d_{c}" for c in CATS]], on=["mo", "t"], how="left")
    tot4 = f4.d_total.fillna(1.0).clip(lower=1e-9)
    cols_g4 = []
    for c in CATS:
        f4[f"share4_{c}"] = f4[f"d_{c}"].fillna(0.0) / tot4
        cols_g4.append(f"share4_{c}")
    f4["log_total4"] = np.log1p(tot4)
    cols_g4.append("log_total4")
    rows += run_g("G4 nation (дефляция индексами РФ)",
                  lambda: window_edges_cols(f4, T_END, pairs, ids, cols_g4),
                  ids, agg_by_id)

    # ---- G1: избыточный YoY-рост в 2024 (статика, 5 осей) ----
    a = pd.read_csv(YOY_RU, sep=";", encoding="utf-8-sig")
    a["ym"] = a.period.str[:7]
    ru_yoy = {}
    for c, cats in RU_YOY_MAP.items():
        sub = a[a.category.isin(cats)]
        ru_yoy[c] = float(sub[sub.ym.between("2024-01", "2024-12")].value.mean()) / 100.0
    g = smoothed_cats().set_index(["mo", "t"])
    exc = {}
    for c in CATS:
        ser = g[f"s_{c}"].unstack("mo")          # index t, cols mo
        yoy = ser / ser.shift(12) - 1
        exc[f"exc_{c}"] = yoy.iloc[12:24].mean().reindex(ids).fillna(0.0) - ru_yoy[c]
    g1 = feats[["mo", "t"] + FEAT_COLS].join(pd.DataFrame(exc).reindex(ids), how="left")
    cols_g1 = FEAT_COLS + [f"exc_{c}" for c in CATS]
    rows += run_g("G1 grow (6dyn + 5 exc_growth, z)",
                  lambda: window_edges_cols(g1, T_END, pairs, ids, cols_g1, z=True),
                  ids, agg_by_id)

    out = pd.DataFrame(rows)
    out.to_csv("out/features_exp4.csv", index=False)
    print(out.to_string(index=False))

    # ---- G3: типология 14 осей ----
    static = pd.read_csv("data/experiments/inputs/derived/mo_static_features.csv").set_index("territory_id")
    stat8 = static[STATIC].copy()
    bfull = bmat.reindex(static.index).to_numpy(float)  # (2548, 24)
    bfull = np.where(np.isnan(bfull), np.nanmedian(bfull, axis=1, keepdims=True), bfull)
    stat8["log_basket"] = np.log1p(bfull.mean(axis=1))
    stat8["growth_basket"] = bfull[:, 12:24].mean(1) / np.clip(bfull[:, 0:12].mean(1), 1e-9, None) - 1
    res, profile, memb = run_typology14(feats, stat8, ids, pairs,
                                        agg_by_id,
                                        stat_cols=STATIC + ["log_basket", "growth_basket"])
    # ядра: топ-3 МО по экономической активности (cons_total_2024 * pop_2024)
    from .features_exp import mo_names
    act = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                      usecols=["territory_id", "cons_total_2024", "pop_2024"])
    act["act"] = act.cons_total_2024.fillna(0) * act.pop_2024.fillna(0)
    act = act.set_index("territory_id")
    names = mo_names()
    for r in profile.itertuples():
        mem_tids = ids[memb == r.cluster]
        top = act.reindex(mem_tids).act.sort_values(ascending=False).head(3)
        profile.loc[r.Index, "cores"] = " — ".join(
            names.get(int(t), str(t)) for t in top.index)
    profile = profile.sort_values("n_mo", ascending=False)
    print("\nG3:", res)
    print(profile[["cluster", "n_mo", "log_basket", "growth_basket", "top3", "cores"]]
          .to_string(index=False))
    profile.to_csv("out/types_profile_g3.csv", index=False)
    # для сравнения — исходная типология (12 осей)
    res12, _, _ = run_typology(feats, static, ids, pairs, agg_by_id)
    print("B12 (контроль):", res12)


if __name__ == "__main__":
    main()
