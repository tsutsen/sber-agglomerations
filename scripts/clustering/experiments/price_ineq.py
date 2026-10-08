"""Ценовая дисперсия регионов: «рынки объединены, но цены не выровнены».

Данные:
- региональный ценовой индекс стандартной корзины (SberIndex, 24 мес 2023-2024)
- потребление МО: per-capita месячные расходы (руб.) x население = суммарно
- рынок/QoL/зарплаты для декомпозиции «почему дорого».

ВАЖНО (единицы): 8_consumption.parquet — СРЕДНИЕ расходы ЖИТЕЛЕЙ МО (руб./чел./мес),
а не региональный оборот. Суммарное годовое потребление региона =
sum(в месяц) value*population. Преминация (coef-1)*расход — верхняя оценка:
индекс корзины применяется ко ВСЕМУ потреблению региона, а не только к корзине.

Результат: out/price_ineq.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parents[3] / "out"
BASKET = OUT.parent / "data/sources/basket_size_per_mo.csv"
LONG = OUT.parent / "data/sources/derived/mo_consumption_monthly_long.csv"
CATS = ["food", "health", "cafe", "transport", "marketplace"]
MON = {n: i + 1 for i, n in enumerate(
    ["январь", "февраль", "март", "апрель", "май", "июнь",
     "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"])}


def main() -> None:
    # 1. Региональный ценовой индекс (средний по 12 мес года)
    b = pd.read_csv(BASKET)
    b["t"] = (b.year - 2023) * 12 + b.month.map(MON) - 1
    c = b.pivot_table(index="region_name", columns="t", values="value").sort_index()
    reg = pd.DataFrame({
        "coef24": c.iloc[:, 12:24].mean(axis=1),
        "coef23": c.iloc[:, 0:12].mean(axis=1),
    })

    # 2. Суммарное годовое потребление региона (per-capita x население)
    lng = pd.read_csv(LONG, usecols=["territory_id", "date", "cat",
                                     "consumption_rub", "population"])
    tot = lng[lng.cat == "total"]

    def annual(year: str) -> pd.Series:
        d = tot[tot.date.str[:4] == year].dropna(subset=["population"])
        d = d.assign(v=d.consumption_rub * d.population)
        return d.groupby("territory_id")["v"].sum()

    a24, a23 = annual("2024"), annual("2023")
    d = pd.read_csv(OUT.parent / "data/sources/derived/mo_consumption_population.csv",
                    usecols=["territory_id", "region_name"])
    d["a24"], d["a23"] = d.territory_id.map(a24), d.territory_id.map(a23)
    r = d.groupby("region_name").agg(a24=("a24", "sum"), a23=("a23", "sum"))
    reg = reg.join(r)
    out: dict = {
        "rf_annual_2024_trl": round(float(reg.a24.sum() / 1e12), 1),
        "rf_annual_2023_trl": round(float(reg.a23.sum() / 1e12), 1),
        "n_regions": int(reg.coef24.notna().sum()),
        "coef24_range": [round(float(reg.coef24.min()), 2), round(float(reg.coef24.max()), 2)],
    }

    # 3. «Преминация»: (index-1)*annual, где index>1
    prem = ((reg.coef24 - 1) * reg.a24).clip(lower=0)
    disc = ((1 - reg.coef24) * reg.a24).clip(lower=0)
    prem23 = ((reg.coef23 - 1) * reg.a23).clip(lower=0)
    out.update(premium_trl=round(float(prem.sum() / 1e12), 2),
               discount_trl=round(float(disc.sum() / 1e12), 2),
               premium_msk_trl=round(float(prem.get("Москва", 0) / 1e12), 2),
               premium_ex_msk_trl=round(float((prem.sum() - prem.get("Москва", 0)) / 1e12), 2),
               premium23_trl=round(float(prem23.sum() / 1e12), 2),
               premium_delta_trl=round(float((prem.sum() - prem23.sum()) / 1e12), 2))
    top5 = (prem.nlargest(5) / 1e9).round(0)
    out["premium_top5_bln"] = top5.to_dict()
    print(f"РФ год: {reg.a24.sum()/1e12:.1f} трлн (2024); "
          f"преминация {prem.sum()/1e12:.2f} трлн (Москва {prem.get('Москва',0)/1e12:.2f}), "
          f"экономия {disc.sum()/1e12:.2f} трлн; "
          f"2023: {prem23.sum()/1e12:.2f} (дельта {(prem.sum()-prem23.sum())/1e12:+.2f})")

    # 4. Динамика ценового уровня
    d_ = (reg.coef24 / reg.coef23 - 1) * 100
    out.update(dyn_median=round(float(d_.median()), 2),
               dyn_p05=round(float(d_.quantile(0.05)), 2),
               dyn_p95=round(float(d_.quantile(0.95)), 2))

    # 5. Корреляции с рынком и QoL
    qol = pd.read_csv(OUT.parent / "data/experiments/inputs/derived/mo_qol.csv",
                      usecols=["territory_id", "year", "income"])
    qinc = qol[qol.year == 2024].set_index("territory_id").income
    mk = pd.read_csv(OUT.parent / "data/experiments/inputs/derived/market_access_per_mo.csv")
    mk = mk.set_index("territory_id")["market_access"]
    j = d[["territory_id", "region_name"]].drop_duplicates("territory_id") \
        .merge(mk.rename("mkt").reset_index(), on="territory_id", how="inner") \
        .merge(reg.reset_index()[["region_name", "coef24"]], on="region_name")
    j["qinc"] = j.territory_id.map(qinc)
    r1 = spearmanr(j.mkt, j.coef24)
    out["mkt_acc_rho"] = round(float(r1.statistic), 3)
    qsub = j.dropna(subset=["qinc"])
    r2 = spearmanr(qsub.qinc, qsub.coef24)
    out["qol_income_rho"] = round(float(r2.statistic), 3)
    out["qol_income_p"] = float(r2.pvalue)
    print(f"5) ценовой уровень vs рынок {r1.statistic:.3f}; "
          f"QoL-доход {r2.statistic:.3f} (p={r2.pvalue:.1e}, n={len(qsub)})")

    # 6. Доля МО региона в кластерах >=3 против ценового уровня
    cl = pd.read_csv(OUT / "clusters_all.csv")
    final = cl[cl.window == cl.window.max()]
    sz = final.groupby("cluster").size()
    bigc = set(sz[sz >= 3].index)
    nmo = d.groupby("region_name").territory_id.nunique()
    fin = final[final.cluster.isin(bigc)].merge(
        d[["territory_id", "region_name"]], left_on="mo", right_on="territory_id")
    regshare = (fin.groupby("region_name").mo.nunique().reindex(reg.index).fillna(0)
                / nmo.reindex(reg.index).fillna(1))
    rr = reg.dropna(subset=["coef24"])
    r4 = spearmanr(regshare.loc[rr.index], rr.coef24)
    out["cluster_share_rho"] = round(float(r4.statistic), 3)

    # 7. Межрегиональные пары <=50 км: production-рёбра vs остальные.
    #    Цены регионального уровня: внутри региона разрыв тождественно 0,
    #    поэтому сравниваем только пары между разными регионами и спрашиваем,
    #    выбирает ли правило «сходный профиль» пары с меньшим ценовым разрывом.
    from .papers_common import load_ctx
    feats, ids, pairs, agg = load_ctx()
    rmap = d.set_index("territory_id").region_name
    coef_arr = rmap.map(reg.coef24).reindex(ids).to_numpy(float)
    e = pd.read_csv(OUT / "edges_2024-12.csv")
    ekeys = set()
    for i, j in zip(e.i, e.j):
        a, b_ = int(i), int(j)
        ekeys.add((min(a, b_), max(a, b_)))
    egap, ngap, n_cand = [], [], 0
    for a, b in pairs:
        a, b = int(a), int(b)
        ta, tb = int(ids[a]), int(ids[b])
        if rmap.loc[ta] == rmap.loc[tb] or pd.isna(coef_arr[a]) or pd.isna(coef_arr[b]):
            continue
        n_cand += 1
        g = abs(float(coef_arr[a]) - float(coef_arr[b]))
        (egap if (min(a, b), max(a, b)) in ekeys else ngap).append(g)
    eg, ng = np.array(egap), np.array(ngap)
    out.update(
        n_cross_pairs=n_cand, n_cross_edges=len(eg),
        cross_edge_gap_lt005=round(float((eg < 0.05).mean()), 3) if len(eg) else None,
        cross_nedge_gap_lt005=round(float((ng < 0.05).mean()), 3),
        cross_edge_gap_lt010=round(float((eg < 0.10).mean()), 3) if len(eg) else None,
        cross_nedge_gap_lt010=round(float((ng < 0.10).mean()), 3),
        cross_edge_gap_median=round(float(np.median(eg)), 3) if len(eg) else None,
        cross_nedge_gap_median=round(float(np.median(ng)), 3),
        # доля пар с cos >= 0.999 среди кандидатов <=50 км (финальное окно)
    )
    from ..network.graphs import FEAT_COLS
    w6 = feats[(feats.t >= feats.t.max() - 5) & (feats.t <= feats.t.max())]
    prof = w6.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0).to_numpy(float)
    nrm = np.linalg.norm(prof, axis=1).clip(min=1e-9)
    ca, cb = pairs[:, 0], pairs[:, 1]
    cosv = (prof[ca] * prof[cb]).sum(1) / (nrm[ca] * nrm[cb])
    out["cos_ge0999_share"] = round(float((cosv >= 0.999).mean()), 3)
    out["n_candidate_pairs"] = int(len(cosv))
    # динамика премии: номинальный рост потребления vs рост доли
    out.update(premium_share23=round(float(prem23.sum() / reg.a23.sum()) * 100, 1),
               premium_share24=round(float(prem.sum() / reg.a24.sum()) * 100, 1))
    print(f"7) межрегион. пар <=50км: {n_cand}, из них рёбер: {len(eg)}; "
          f"gap<5%: {len(eg) and (eg < 0.05).mean():.3f} vs {(ng < 0.05).mean():.3f}; "
          f"gap<10%: {len(eg) and (eg < 0.10).mean():.3f} vs {(ng < 0.10).mean():.3f}; "
          f"cos>=0.999: {(cosv >= 0.999).mean():.3f}")

    # 8. Декомпозиция «почему дорого»: зарплаты, расстояние до ближайшего
    #    мегаполиса (логистика, 31 хаб >1 млн), доступ к рынку. Не Москва:
    #    Москва — одновременно аутлайер премии и один из хабов.
    from scipy.spatial import distance as sdist
    from .papers_common import coords
    lat, lon = coords(ids)
    pos = {int(t): k for k, t in enumerate(ids)}
    pop = pd.read_csv(OUT.parent / "data/sources/population_per_mo.csv")
    ptot = pop[pop.age == "Всего"].groupby("territory_id").value.sum()
    hubs_ptot = ptot[ptot > 1_000_000]
    hubidx = [pos[int(t)] for t in hubs_ptot.index if int(t) in pos]
    hub = np.radians(np.column_stack([lon[hubidx], lat[hubidx]]))
    g = np.radians(np.column_stack([lon, lat]))
    dhub = (6371 * sdist.cdist(g, hub)).min(axis=1)
    static = pd.read_csv(OUT.parent / "data/experiments/inputs/derived/mo_static_features.csv").set_index("territory_id")
    base = d.assign(log_salary=d.territory_id.map(static.log_salary),
                    dist_hub=d.territory_id.map(pd.Series(dhub, index=ids)),
                    mkt=d.territory_id.map(mk)).dropna()
    rg = base.groupby("region_name").agg(
        log_salary=("log_salary", "median"), dist_hub=("dist_hub", "median"),
        mkt=("mkt", "median")).join(reg)
    dec = {}
    for col in ("log_salary", "dist_hub", "mkt"):
        rr = spearmanr(rg[col], rg.coef24)
        dec[col] = round(float(rr.statistic), 3)
    rg2 = rg.drop(index="Москва")
    dec_ex = {col: round(float(spearmanr(rg2[col], rg2.coef24).statistic), 3)
              for col in ("log_salary", "dist_hub")}
    out.update(decomp=dec, decomp_ex_msk=dec_ex, n_hubs=len(hubidx))
    print(f"8) декомпозиция: {dec} (без Москвы: {dec_ex}); хабов: {len(hubidx)}")

    # 9. Премия по кластерам (финальное окно): premium = sum((coef-1)*потребление МО),
    #    только «дорогая» часть; + внутренние разброс цен внутри кластера.
    cl = final.merge(d[["territory_id", "region_name"]],
                     left_on="mo", right_on="territory_id", how="left")
    cl["coef"] = cl.region_name.map(reg.coef24)
    cl["a24"] = cl.mo.map(a24)
    cl["prem"] = ((cl.coef - 1) * cl.a24).clip(lower=0)
    nn = pd.read_csv(OUT.parent / "data/experiments/inputs/derived/cluster_names.csv")
    cl["name"] = cl.cluster.map(nn.set_index("cluster").name)
    cl["region_mode"] = cl.groupby("cluster")["region_name"].transform(
        lambda s: s.value_counts().index[0])
    g = cl.groupby("cluster").agg(
        size=("mo", "nunique"),
        regions=("region_name", "nunique"),
        premium_bln=("prem", "sum"),
        price_gap=("coef", lambda s: (s.max() - s.min())),
        name=("name", "first"),
        region=("region_mode", "first"))
    g["premium_bln"] = (g.premium_bln / 1e9).round(1)
    g["price_gap"] = g.price_gap.round(3)
    g = g.sort_values("premium_bln", ascending=False)
    out["cluster_premium_total_bln"] = round(float(g.premium_bln.sum()), 0)
    out["n_clusters_cross_region"] = int((g.regions > 1).sum())
    out["n_clusters_ge5pct_gap"] = int((g.price_gap >= 0.05).sum())
    # топ-5 регионов по приросту премии 23->24 (млрд)
    dprem = ((prem - prem23) / 1e9).sort_values(ascending=False)
    out["premium_growth_top5"] = {
        str(k): round(float(v), 0) for k, v in dprem.head(5).items()}
    out["premium_growth_ex_msk_bln"] = round(float(dprem.iloc[1:].sum()), 0)
    print(f"9) кластеров: {len(g)}, сумм. премия {g.premium_bln.sum():.0f} млрд; "
          f">1 региона: {(g.regions > 1).sum()}; разброс >=5%: {(g.price_gap >= 0.05).sum()}")
    print(g.head(8).to_string())

    # Диаметр кластеров — фильтр артефактов-коридоров.
    from .papers_common import coords, haversine
    lat, lon = coords(ids)
    Dm = haversine(lat, lon)
    diams = {}
    for c, sub in final.groupby("cluster"):
        idx = [pos[int(mo)] for mo in sub.mo]
        diams[c] = round(float(Dm[np.ix_(idx, idx)].max()) if len(idx) > 1 else 0.0, 1)
    g["diam_km"] = g.index.map(diams)
    g.to_csv(OUT / "cluster_premium.csv", encoding="utf-8-sig")
    out["n_cl_diam_gt500"] = int((g[g["size"] >= 3].diam_km > 500).sum())
    out["prem_bln_cl_diam_gt500"] = round(
        float(g[(g["size"] >= 3) & (g.diam_km > 500)].premium_bln.sum()), 0)
    out["cluster_premium_top15"] = (g.head(15).reset_index().to_dict("records"))

    OUT.mkdir(exist_ok=True)
    with open(OUT / "price_ineq.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"-> {OUT / 'price_ineq.json'}")


if __name__ == "__main__":
    main()
