"""EXP 4: внутренние индексы качества с null-базами (Shalileh 2025, Gagolewski 2022).

Пространство признаков (X, 14 осей; размещение = типология):
  SW (силуэт), CH (Calinski–Harabasz), S_Dbw (Davies–Bouldin) + null:
  100 случайных размещений тех же размеров.
Пространство сети (Shalileh 2025, их eq. 18-22):
  AVI (average isolability) = средняя доля веса кластера, уходящая внутрь;
  AVU (average unifiability) = средняя доля веса кластера, уходящая в
  каждый другой; ANUI = AVI / (1 + AVI*AVU).
  Считаем для (а) production-графа (3637 рёбер) и (б) плотной
  косинус-14 матрицы (типология) + null случайных размещений тех же
  размеров.

Важно (Gagolewski 2022): внутренние индексы — критерии ОЦЕНКИ, а не
цели оптимизации; мы их так и используем (оптимизировали по purity/ARI
на внешнем референсе, а не по SW/CH).

Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.paper_exp4_cvi.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

from .papers_common import (load_ctx, prod_edges, prod_members, typ14)

N_NULL = 100
SEED = 11


def avi_avu(A: np.ndarray, members: np.ndarray) -> tuple[float, float, float]:
    """eq. 18-22 статьи: изолируемость/унифицируемость по весам ребёр.
    Через one-hot: W_kl = S_k^T A S_l (одна матричное умножение)."""
    K = int(members.max()) + 1
    S = np.zeros((len(members), K))
    S[np.arange(len(members)), members] = 1.0
    W = S.T @ A @ S            # W[k, l] = вес между k и l
    tot = (A.sum(axis=1) @ S)  # tot[k] = суммарный вес узлов k
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.divide(W, tot[:, None], out=np.zeros_like(W), where=tot[:, None] > 0)
    np.fill_diagonal(frac, 0.0)
    iso = np.divide(np.diag(W), tot, where=tot > 0, out=np.zeros(K))
    avi = float(iso.mean())
    avu = float(frac.sum() / (K * (K - 1)))
    anui = avi / (1 + avi * avu)
    return avi, avu, anui


def random_partitions(members: np.ndarray, rng: np.random.Generator,
                      n: int = N_NULL) -> list[np.ndarray]:
    sizes = np.array([(members == k).sum() for k in np.unique(members)])
    out = []
    for _ in range(n):
        lab = np.empty(len(members), dtype=int)
        order = rng.permutation(len(members))
        pos, nid = 0, 0
        for s in sizes:
            lab[order[pos:pos + int(s)]] = nid
            pos += int(s)
            nid += 1
        out.append(lab)
    return out


def main() -> None:
    feats, ids, pairs, _ = load_ctx()
    rng = np.random.default_rng(SEED)

    # ---------- пространство признаков: типология ----------
    X, cols, mt, q = typ14(feats, ids, pairs)
    sw = float(silhouette_score(X, mt))
    ch = float(calinski_harabasz_score(X, mt))
    dbw = float(davies_bouldin_score(X, mt))
    rands = random_partitions(mt, rng)
    sw_r = [float(silhouette_score(X, r)) for r in rands]
    ch_r = [float(calinski_harabasz_score(X, r)) for r in rands]
    dbw_r = [float(davies_bouldin_score(X, r)) for r in rands]

    def stat(obs: float, vals: list[float], higher: bool) -> dict:
        best = float(np.max(vals) if higher else np.min(vals))
        p = float(np.mean(np.asarray(vals) >= obs if higher
                          else np.asarray(vals) <= obs))
        return {"obs": round(obs, 3), "null_mean": round(float(np.mean(vals)), 3),
                "null_best": round(best, 3), "p": round(p, 3)}

    rows = [
        {"space": "features(14)", "partition": "typology", "metric": "SW",
         **stat(sw, sw_r, True)},
        {"space": "features(14)", "partition": "typology", "metric": "CH",
         **stat(ch, ch_r, True)},
        {"space": "features(14)", "partition": "typology", "metric": "S_Dbw",
         **stat(dbw, dbw_r, False)},
    ]

    # ---------- пространство сети ----------
    # (а) production-граф
    E = prod_edges()
    n = len(ids)
    A = np.zeros((n, n))
    ii, jj, ww = E[:, 0].astype(int), E[:, 1].astype(int), E[:, 2]
    A[ii, jj] = ww
    A[jj, ii] = ww
    m = prod_members(ids)
    a_obs, u_obs, anui_obs = avi_avu(A, m)
    vals = [avi_avu(A, r) for r in random_partitions(m, rng)]
    avi_r = [v[0] for v in vals]
    avu_r = [v[1] for v in vals]
    anui_r = [v[2] for v in vals]
    rows.append({"space": "network(prod)", "partition": "production",
                 "metric": "AVI (isolability)",
                 **stat(a_obs, avi_r, True)})
    rows.append({"space": "network(prod)", "partition": "production",
                 "metric": "AVU (unifiability)",
                 **stat(u_obs, avu_r, False)})
    rows.append({"space": "network(prod)", "partition": "production",
                 "metric": "ANUI", **stat(anui_obs, anui_r, True)})

    # (б) плотная косинус-14, типология
    norm = np.linalg.norm(X, axis=1).clip(min=1e-9)
    At = np.clip((X / norm[:, None]) @ (X / norm[:, None]).T, 0.0, None)
    np.fill_diagonal(At, 0.0)
    a_t, u_t, anui_t = avi_avu(At, mt)
    vals_t = [avi_avu(At, r) for r in rands]
    rows.append({"space": "network(cos14)", "partition": "typology",
                 "metric": "AVI (isolability)",
                 **stat(a_t, [v[0] for v in vals_t], True)})
    rows.append({"space": "network(cos14)", "partition": "typology",
                 "metric": "AVU (unifiability)",
                 **stat(u_t, [v[1] for v in vals_t], False)})
    rows.append({"space": "network(cos14)", "partition": "typology",
                 "metric": "ANUI", **stat(anui_t, [v[2] for v in vals_t], True)})

    out = pd.DataFrame(rows)
    out.to_csv("out/paper_exp4_cvi.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
