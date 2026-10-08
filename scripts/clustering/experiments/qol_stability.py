"""Стабильность типологии по Индексу качества жизни (2025 как out-of-time год).

Типология (G3, 14 осей, 15 типов) построена на фичах 2024 по всем 2144 МО.
Проверяем на подмножестве МО-городов (234-243), у которых есть QoL за 2024 и 2025:
  1. Сепарация типов в 2025-пространстве: silhouette (CORE 8, z) + ANOVA/Kruskal
     по ikz8, против null (перестановки лэйблов).
  2. Стабильность профилей: корреляция Спирмена средних профилей CORE
     2024 vs 2025 по типам (типы >= 5 городов) + ранковая корреляция ikz8.
  3. ARI пересборки: тот же пайплайн (cos + mutual kNN + Louvain) на CORE-2024
     и CORE-2025; ARI между двумя разбиениями городского подмножества.

Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.qol_stability.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal, spearmanr
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .papers_common import load_ctx, typ14
from .features_exp import zscore
from ..network.graphs import KNN, louvain

CORE = ["housing", "income", "health", "education",
        "mobility", "environment", "ecology", "safety"]
SEED = 42


def knn_louvain(X: np.ndarray, k: int = KNN):
    """Копия пайплайна typ14: косинус + mutual kNN + Louvain -> labels."""
    norm = np.linalg.norm(X, axis=1).clip(min=1e-9)
    sim = (X / norm[:, None]) @ (X / norm[:, None]).T
    np.fill_diagonal(sim, -1.0)
    sim = np.where(sim > 0, sim, -1.0)
    idx = np.argsort(-sim, axis=1)[:, :k]
    iu = np.repeat(np.arange(len(X)), k)
    ju = idx.ravel()
    w = sim[iu, ju]
    ok = w > 0
    lo, hi = np.minimum(iu[ok], ju[ok]), np.maximum(iu[ok], ju[ok])
    uniq = dict(zip(zip(lo, hi), w[ok], strict=True))
    edges = [(int(i), int(j), float(v)) for (i, j), v in uniq.items()]
    members, _ = louvain(len(X), edges)
    return members


def main() -> None:
    feats, ids, pairs, _ = load_ctx()
    X, cols, members, q = typ14(feats, ids, pairs)
    print(f"типология: {len(np.unique(members))} типов, q={q:.3f} (2024, все МО)")

    qol = pd.read_csv("data/experiments/inputs/derived/mo_qol.csv")
    # МО с полными CORE за оба года
    def full(year: int) -> set[int]:
        d = qol[qol.year == year].set_index("territory_id")
        return set(d.index[d[CORE].notna().all(axis=1)])

    both = sorted(full(2024) & full(2025))
    pos = {int(t): k for k, t in enumerate(ids)}
    sel = [t for t in both if t in pos]
    lab = np.array([members[pos[t]] for t in sel])
    d24 = qol[qol.year == 2024].set_index("territory_id").loc[sel]
    d25 = qol[qol.year == 2025].set_index("territory_id").loc[sel]
    v24, v25 = d24[CORE].to_numpy(float), d25[CORE].to_numpy(float)
    ik24, ik25 = d24["ikz8"].to_numpy(float), d25["ikz8"].to_numpy(float)
    Z24, Z25 = zscore(v24), zscore(v25)
    types = np.unique(lab)
    cnt = np.bincount(lab, minlength=lab.max() + 1)
    print(f"городов с QoL 2024+2025: {len(sel)}; типов: {len(types)}; "
          f"типов с >=5 городов: {int((cnt[cnt >= 1] >= 5).sum())}")

    out: dict = {"n_cities": len(sel), "n_types": int(len(types))}

    # 1. Сепарация в 2025-пространстве
    sil = silhouette_score(Z25, lab)
    rng = np.random.default_rng(SEED)
    null = [silhouette_score(Z25, rng.permutation(lab)) for _ in range(100)]
    fstat, fp = f_oneway(*[ik25[lab == t] for t in types if (lab == t).sum() >= 5])
    kstat, kp = kruskal(*[ik25[lab == t] for t in types if (lab == t).sum() >= 5])
    out.update(sil_2025=sil, sil_null_max=float(max(null)),
               anova_F=float(fstat), anova_p=float(fp),
               kruskal_p=float(kp))
    print(f"1) silhouette(2025, типы 2024) = {sil:.3f} (null max {max(null):.3f}); "
          f"ANOVA F={fstat:.1f} p={fp:.1e}; Kruskal p={kp:.1e}")

    # 2. Стабильность профилей
    cells_y, cells_x = [], []
    for t in types:
        msk = lab == t
        if msk.sum() < 5:
            continue
        p24, p25 = v24[msk].mean(0), v25[msk].mean(0)
        for c in CORE:
            cells_y.append(p24[CORE.index(c)])
            cells_x.append(p25[CORE.index(c)])
    rho, p = spearmanr(cells_y, cells_x)
    ik_rho, ik_p = spearmanr(ik24, ik25)
    out.update(profile_rho=float(rho), profile_p=float(p),
               ikz8_rho=float(ik_rho), ikz8_p=float(ik_p))
    print(f"2) профиль CORE 2024 vs 2025: Спирмен {rho:.3f} (p={p:.1e}) "
          f"по {len(cells_y)//8} типам x 8 направлений; "
          f"ikz8 по городам: {ik_rho:.3f} (p={ik_p:.1e})")

    # 3. ARI пересборки на CORE
    l24 = knn_louvain(Z24)
    l25 = knn_louvain(Z25)
    ari = adjusted_rand_score(l24, l25)
    out.update(ari_rebuild=float(ari),
               n_clusters_2024=int(len(np.unique(l24))),
               n_clusters_2025=int(len(np.unique(l25))))
    print(f"3) ARI пересборки kNN+Louvain: CORE-2024 vs CORE-2025 = {ari:.3f} "
          f"({len(np.unique(l24))} vs {len(np.unique(l25))} кластеров)")

    with open("out/qol_stability.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("-> out/qol_stability.json")


if __name__ == "__main__":
    main()
