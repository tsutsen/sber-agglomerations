"""Эксперимент: расширенное пространство признаков (6 динамических + 6 статических).

Варианты (финальное окно t_end=23, 12/2024, окно 6 мес):
  prod     — текущее производство: косинус по 6 динамическим, без z-score
             (правило: <=50 км + топ-8, симметризация).
  A        — ext_std: то же правило, косинус по 12 признакам, все z-scored.
  C        — base_6_std: то же правило, старые 6 признаков, z-scored (абляция).
  B        — типология локальных экономик: МО-вектор [среднее 6 динамических
             за 24 мес + 6 статических], z-scored, mutual kNN топ-8 (БЕЗ
             расстояния) -> Louvain. Плюс пространственная когерентность
             против 20 перемешанных размечений (p-value).
  D        — year-over-year: production-правило в окнах t_end=11 (09/2023) и
             t_end=23 (12/2024) — НЕперекрывающиеся; доля МО, сохранивших
             кластер, и best-match Jaccard.
  E1       — sub8: 6 динамических + log_salary + log_market_acc, z-scored.
  E2       — raw12: 12 признаков без z-score (отделяет эффект стандартизации).
  E3       — Leiden (igraph) на production-рёбрах вместо Louvain.
  E4       — DTW: сходство 1/(1+DTW/2T) по 24-мес. 6-мерным рядам,
             то же правило <=50 км + топ-8 (лаг/фаза вместо синхронного косинуса).

Порог выбора (объявлен заранее): A или C становится производством, только если
ARI >= +0.05 ИЛИ purity >= +0.03 к текущему, при сопоставимом распределении типов.

Пишет:  out/features_exp.csv, out/types_profile.csv
Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.features_exp.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from ..features import FEAT_COLS, load_features
from ..network.graphs import D_CUTOFF, KNN, WINDOW, candidate_pairs, louvain, window_edges
from ..tune import purity

T_END = 23          # 12/2024
T_END_YOY = 11      # 09/2023 (неперекрывающееся окно)
STATIC = ["log_salary", "work_share", "mig_per_1000",
          "log_market_acc", "cons_growth", "season_amp"]


def zscore(X: np.ndarray) -> np.ndarray:
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def window_edges_cols(feats: pd.DataFrame, t_end: int, pairs: np.ndarray, ids: np.ndarray,
                      cols: list[str], k: int = KNN, window: int = WINDOW,
                      z: bool = False) -> list[tuple[int, int, float]]:
    """Локальная версия network.graphs.window_edges с параметром cols и z-score."""
    w = feats[(feats.t >= t_end - window + 1) & (feats.t <= t_end)]
    prof = w.groupby("mo")[cols].mean()
    prof = prof.reindex(ids).fillna(0.0).to_numpy(dtype=float)
    if z:
        prof = zscore(prof)
    norm = np.linalg.norm(prof, axis=1).clip(min=1e-9)
    a, b = pairs[:, 0], pairs[:, 1]
    cos = (prof[a] * prof[b]).sum(1) / (norm[a] * norm[b])
    # сохраняем w >= 0: нулевые сходства — заглушки изолированных МО
    # (совпадает с production window_edges); отрицательные (z-scored) — не связь
    cos = np.where(cos >= 0, cos, np.nan)
    df = pd.DataFrame({"i": a, "j": b, "w": cos}).dropna(subset=["w"])
    both = pd.concat([df, df.assign(i=df.j, j=df.i)], ignore_index=True)
    both = both.sort_values("w", ascending=False).groupby("i").head(k)
    lo, hi = np.minimum(both.i, both.j), np.maximum(both.i, both.j)
    uniq = dict(zip(zip(lo, hi), both.w.to_numpy(), strict=True))
    return [(int(i), int(j), float(w)) for (i, j), w in uniq.items()]


def ari_all(members: np.ndarray, ids: np.ndarray, agg_by_id: dict) -> float:
    """ARI по ВСЕМ МО официального перечня (кластерный лэйбл vs агломерация)."""
    mask = np.array([int(t) in agg_by_id for t in ids])
    y = np.array([agg_by_id[int(t)] for t in ids[mask]])
    return float(adjusted_rand_score(y, members[mask]))


def run_variant(name: str, feats: pd.DataFrame, t_end: int, pairs: np.ndarray,
                ids: np.ndarray, cols: list[str], z: bool,
                agg_by_id: dict) -> dict:
    edges = window_edges_cols(feats, t_end, pairs, ids, cols, z=z)
    members, q = louvain(len(ids), edges)
    return {"variant": name, "edges": len(edges), "clusters": int(members.max() + 1),
            "clusters_3plus": int(sum((members == c).sum() >= 3 for c in np.unique(members))),
            "q": round(float(q), 3),
            "purity": round(purity(members, ids, agg_by_id), 3),
            "ari": round(ari_all(members, ids, agg_by_id), 3)}


def leiden(n: int, edges: list[tuple[int, int, float]]) -> tuple[np.ndarray, float]:
    import igraph as ig
    g = ig.Graph(n, [(e[0], e[1]) for e in edges])
    g.es["weight"] = [e[2] for e in edges]
    cl = g.community_leiden(weights="weight")
    return np.asarray(cl.membership), float(cl.q)


def dtw_batch(Xa: np.ndarray, Xb: np.ndarray) -> np.ndarray:
    """DTW по парам: Xa, Xb — (n, T, D); возвращает (n,) дистанцию (сумма квадратов)."""
    n, T, D = Xa.shape
    cost = ((Xa[:, :, None, :] - Xb[:, None, :, :]) ** 2).sum(-1)  # (n, T, T)
    dist = np.full((n, T, T), np.inf)
    dist[:, 0, 0] = cost[:, 0, 0]
    for i in range(T):
        for j in range(T):
            if i == 0 and j == 0:
                continue
            m = np.full(n, np.inf)
            if i > 0:
                m = np.minimum(m, dist[:, i - 1, j])
            if j > 0:
                m = np.minimum(m, dist[:, i, j - 1])
            if i > 0 and j > 0:
                m = np.minimum(m, dist[:, i - 1, j - 1])
            dist[:, i, j] = m + cost[:, i, j]
    return dist[:, -1, -1]


def run_typology(feats: pd.DataFrame, static: pd.DataFrame, ids: np.ndarray,
                 pairs: np.ndarray, agg_by_id: dict) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """B: типология на 24-мес. среднем (12 осей, z-scored), mutual kNN без расстояния."""
    dyn = feats.groupby("mo")[FEAT_COLS].mean().reindex(ids).fillna(0.0)
    stat = static[STATIC].reindex(ids).fillna(0.0)
    X = zscore(np.hstack([dyn.to_numpy(dtype=float), stat.to_numpy(dtype=float)]))
    cols12 = FEAT_COLS + STATIC
    norm = np.linalg.norm(X, axis=1).clip(min=1e-9)
    sim = (X / norm[:, None]) @ (X / norm[:, None]).T
    np.fill_diagonal(sim, -1.0)
    sim = np.where(sim > 0, sim, -1.0)  # топ-k берём только среди положительных
    idx = np.argsort(-sim, axis=1)[:, :KNN]
    iu = np.repeat(np.arange(len(ids)), KNN); ju = idx.ravel()
    w = sim[iu, ju]
    ok = w > 0
    lo, hi = np.minimum(iu[ok], ju[ok]), np.maximum(iu[ok], ju[ok])
    uniq = dict(zip(zip(lo, hi), w[ok], strict=True))
    edges = [(int(i), int(j), float(w)) for (i, j), w in uniq.items()]
    members, q = louvain(len(ids), edges)

    # пространственная когерентность: доля МО с соседем своего кластера <=50 км
    def coherence(m: np.ndarray) -> float:
        nb = {}
        for a, b in zip(*pairs.T):
            nb.setdefault(int(a), set()).add(int(b))
            nb.setdefault(int(b), set()).add(int(a))
        ok = 0
        for i in range(len(m)):
            if any(m[j] == m[i] for j in nb.get(i, ())):
                ok += 1
        return ok / len(m)

    coh = coherence(members)
    rng = np.random.default_rng(42)
    shuffled = [coherence(rng.permutation(members)) for _ in range(20)]
    pval = (sum(s >= coh for s in shuffled) + 1) / (20 + 1)

    res = {"variant": "B_typology", "edges": len(edges), "clusters": int(members.max() + 1),
           "clusters_5plus": int(sum((members == c).sum() >= 5 for c in np.unique(members))),
           "clusters_10plus": int(sum((members == c).sum() >= 10 for c in np.unique(members))),
           "q": round(float(q), 3),
           "spatial_coh": round(coh, 3),
           "spatial_coh_null_max": round(max(shuffled), 3),
           "p_value": round(pval, 3),
           "ari": round(ari_all(members, ids, agg_by_id), 3)}

    # профиль: средние z по 12 признакам, кластеры >=5 МО
    rows = []
    for cid in np.unique(members):
        mem = members == cid
        if mem.sum() < 5:
            continue
        z = X[mem].mean(0)
        top = np.argsort(-np.abs(z))[:3]
        rows.append({"cluster": int(cid), "n_mo": int(mem.sum()),
                     **{c: round(float(z[j]), 2) for j, c in enumerate(cols12)},
                     "top3": " > ".join(cols12[j] for j in top)})
    profile = pd.DataFrame(rows).sort_values("n_mo", ascending=False)
    return res, profile, members


def run_yoy(feats: pd.DataFrame, pairs: np.ndarray, ids: np.ndarray) -> dict:
    """D: production-правило в неперекрывающихся окнах t_end=11 и t_end=23."""
    m11, _ = louvain(len(ids), window_edges(feats, T_END_YOY, pairs, ids))
    m23, _ = louvain(len(ids), window_edges(feats, T_END, pairs, ids))
    c11 = {c: set(np.where(m11 == c)[0]) for c in np.unique(m11)}
    kept, jacs = 0, []
    for c in np.unique(m23):
        s23 = set(np.where(m23 == c)[0])
        best, bj = None, -1.0
        for c2, s11 in c11.items():
            j = len(s23 & s11) / len(s23 | s11)
            if j > bj:
                best, bj = c2, j
        jacs.append(bj)
        if bj >= 0.5:  # кластер узнаваем: >50% состава сохранилось
            kept += len(s23)
    jacs = np.array(jacs)
    return {"t11_clusters": len(c11), "t23_clusters": int(m23.max() + 1),
            "mo_share_kept": round(kept / len(m23), 3),
            "best_jaccard_mean": round(float(jacs.mean()), 3),
            "best_jaccard_median": round(float(np.median(jacs)), 3)}


def mo_names() -> dict:
    from .cluster_names import clean
    mo = pd.read_csv("data/experiments/inputs/derived/mo_names.csv")
    return {int(r.territory_id): clean(r.municipal_district_name)
            for r in mo.itertuples() if pd.notna(r.municipal_district_name)}


def main() -> None:
    feats, _ = load_features()
    ids = np.sort(feats.mo.unique())
    off = pd.read_csv("data/experiments/inputs/derived/official_agglomerations_mapped.csv")
    off = off[off.territory_id.notna()]  # type: ignore[truthy-function]
    agg_by_id = dict(zip(off.territory_id, off["Агломерация"], strict=True))
    pairs, _ = candidate_pairs("data/sources", ids)
    static = pd.read_csv("data/experiments/inputs/derived/mo_static_features.csv").set_index("territory_id")

    # расширяем feats статическими колонками (одинаковы по месяцам);
    # все МО из feats покрываются static (медианная импутация там уже сделана)
    feats12 = feats.merge(static.reset_index().rename(columns={"territory_id": "mo"}),
                          on="mo", how="left")
    assert feats12[STATIC].notna().all().all(), "статические признаки не покрылись"

    rows = [
        run_variant("prod (6 dyn, raw)", feats, T_END, pairs, ids, FEAT_COLS, z=False,
                    agg_by_id=agg_by_id),
        run_variant("A (12, z)", feats12, T_END, pairs, ids, FEAT_COLS + STATIC, z=True,
                    agg_by_id=agg_by_id),
        run_variant("C (6 dyn, z)", feats, T_END, pairs, ids, FEAT_COLS, z=True,
                    agg_by_id=agg_by_id),
    ]
    act = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                      usecols=["territory_id", "cons_total_2024", "pop_2024"])
    act["act"] = act.cons_total_2024.fillna(0) * act.pop_2024.fillna(0)
    act = act.set_index("territory_id")
    names = mo_names()

    # E-варианты
    rows.append(run_variant("E1 (8, z)", feats12, T_END, pairs, ids,
                            FEAT_COLS + ["log_salary", "log_market_acc"], z=True,
                            agg_by_id=agg_by_id))
    rows.append(run_variant("E2 (12, raw)", feats12, T_END, pairs, ids,
                            FEAT_COLS + STATIC, z=False, agg_by_id=agg_by_id))
    pe = window_edges(feats, T_END, pairs, ids)
    e3m, e3q = leiden(len(ids), pe)
    rows.append({"variant": "E3 (leiden, prod edges)", "edges": len(pe), "clusters": int(e3m.max() + 1),
                 "clusters_3plus": int(sum((e3m == c).sum() >= 3 for c in np.unique(e3m))),
                 "q": round(float(e3q), 3),
                 "purity": round(purity(e3m, ids, agg_by_id), 3),
                 "ari": round(ari_all(e3m, ids, agg_by_id), 3)})

    # E4: DTW по 24-мес. 6-мерным рядам
    pv = feats.pivot_table(index="mo", columns="t", values=FEAT_COLS).reindex(ids)
    X = pv.to_numpy(dtype=float)
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    X = X.reshape(len(ids), -1, len(FEAT_COLS))
    a, b = pairs[:, 0], pairs[:, 1]
    sims = np.empty(len(pairs))
    B = 2048
    for s0 in range(0, len(pairs), B):
        sl = slice(s0, s0 + B)
        d = dtw_batch(X[a[sl]], X[b[sl]])
        sims[sl] = 1.0 / (1.0 + np.sqrt(d) / 48.0)
    # топ-8 симметрично, как в window_edges
    df = pd.DataFrame({"i": a, "j": b, "w": sims})
    both = pd.concat([df, df.assign(i=df.j, j=df.i)], ignore_index=True)
    both = both.sort_values("w", ascending=False).groupby("i").head(KNN)
    lo, hi = np.minimum(both.i, both.j), np.maximum(both.i, both.j)
    uniq = dict(zip(zip(lo, hi), both.w.to_numpy(), strict=True))
    e4m, e4q = louvain(len(ids), [(int(i), int(j), float(w)) for (i, j), w in uniq.items()])
    rows.append({"variant": "E4 (DTW 24m)", "edges": len(uniq),
                 "clusters": int(e4m.max() + 1),
                 "clusters_3plus": int(sum((e4m == c).sum() >= 3 for c in np.unique(e4m))),
                 "q": round(float(e4q), 3),
                 "purity": round(purity(e4m, ids, agg_by_id), 3),
                 "ari": round(ari_all(e4m, ids, agg_by_id), 3)})

    # разбег по запуску (production, без seed): 3 прогона
    sp = []
    for _ in range(3):
        em, _ = louvain(len(ids), window_edges(feats, T_END, pairs, ids))
        sp.append((purity(em, ids, agg_by_id), ari_all(em, ids, agg_by_id)))
    sp = np.array(sp)
    print(f"\nРазбег production (3 прогона без seed): purity {sp[:,0].min():.3f}-{sp[:,0].max():.3f}, "
          f"ARI {sp[:,1].min():.3f}-{sp[:,1].max():.3f}")

    # типология B (один раз): метрики + профиль + ядра (топ-3 МО по активности)
    res_b, profile, memb_b = run_typology(feats, static, ids, pairs, agg_by_id)
    rows.append(res_b)
    for r in profile.itertuples():
        mem_tids = ids[memb_b == r.cluster]
        top = act.reindex(mem_tids).act.sort_values(ascending=False).head(3)
        profile.loc[r.Index, "cores"] = " — ".join(names.get(int(t), str(t)) for t in top.index)
    profile.to_csv("out/types_profile.csv", index=False)

    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))
    grid.to_csv("out/features_exp.csv", index=False)
    print("\nYoY (D), production-правило, неперекрывающиеся окна 09/2023 -> 12/2024:")
    yoy = run_yoy(feats, pairs, ids)
    for k, v in yoy.items():
        print(f"  {k}: {v}")
    pd.DataFrame([yoy]).to_csv("out/features_exp_yoy.csv", index=False)
    print(f"\nКандидатские пары <= {D_CUTOFF} км: {len(pairs)}; МО: {len(ids)}")
    print("профиль типов (B) -> out/types_profile.csv")
    print(profile.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
