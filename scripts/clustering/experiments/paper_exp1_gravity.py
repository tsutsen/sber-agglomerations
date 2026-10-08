"""EXP 1: пространственная модулярность с гравитационным null (Expert et al., PNAS 2011).

Идея статьи: в пространственных сетях стандартная модулярность (NG-null)
находит тривиально компактные модули; если заменить null на гравитационный
  P_ij = N_i * N_j * f(d_ij),  f(d) = Σ A_ij / Σ N_iN_j  (по расстоянию d),
то Q_Spa вознаграждает связи, которые сильнее гравитационного ожидания на
этом расстоянии, — т.е. структуру, не объяснимую просто близостью.

Наша проверка (для production-размещения и для типологии):
  Q_Spa(наблюд. веса, наблюд. позиции)
  Q_Spa(переставленные рёбра, те же веса)       <- 100 реализаций
  Q_Spa(наблюд. веса, перемешанные позиции)     <- 100 перестановок координат
  + Q_NG (стандартная модулярность) для справки.

Интерпретация (табл. 1 статьи): z_позиц << 0 — размещение пространственно
компактное, гравитация его объясняет; z_рёбра >> 0 — структура сильно
сверх гравитационного ожидания (ненулевая, не-пространственная).
Позиции = центроиды МО, расстояния — гаверсинус (и в наблюдении, и в null —
чистое сравнение; дорожные расстояния matrix_distance_mo_pairs пропорциональны).

Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.paper_exp1_gravity.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .papers_common import (coords, haversine, load_ctx, population,
                                   prod_edges, prod_members, typ14)
from .features_exp import ari_all

N_NULL = 100
SEED = 7


def bin_ids(D: np.ndarray) -> np.ndarray:
    """Кусочная биновка расстояний: 1 км до 100, 10 км до 1000, 100 км выше."""
    b = np.empty_like(D, dtype=int)
    m100 = D < 100
    m1000 = (D >= 100) & (D < 1000)
    rest = D >= 1000
    b[m100] = np.floor(D[m100]).astype(int)
    b[m1000] = 100 + np.floor(D[m1000] / 10).astype(int)
    b[rest] = 200 + np.floor(D[rest] / 100).astype(int)
    return b


def q_of(A: np.ndarray, P: np.ndarray, members: np.ndarray) -> float:
    m = A.sum() / 2
    same = members[:, None] == members[None, :]
    return float((((A - P) * same).sum()) / (2 * m))


def q_ng(A: np.ndarray, members: np.ndarray) -> float:
    m = A.sum() / 2
    k = A.sum(1)
    P = np.outer(k, k) / (2 * m)
    same = members[:, None] == members[None, :]
    return float((((A - P) * same).sum()) / (2 * m))


def evaluate(A: np.ndarray, N: np.ndarray, D: np.ndarray, B: np.ndarray,
             lat: np.ndarray, lon: np.ndarray, members: np.ndarray,
             rng: np.random.Generator) -> dict:
    mass = (N[:, None] * N[None, :])
    num = np.bincount(B.ravel(), weights=A.ravel(), minlength=B.max() + 1)
    den = np.bincount(B.ravel(), weights=mass.ravel(), minlength=B.max() + 1)
    f = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    P = mass * f[B]
    q_obs = q_of(A, P, members)

    # перестановка концов рёбер: те же веса, случайные пары (случайная сеть
    # с тем же набором весов; для плотной матрицы — то же самое, что
    # перемешивание весов между парами)
    idx = np.argwhere(A > 0)
    w_obs = A[idx[:, 0], idx[:, 1]]
    n = len(members)
    q_w = []
    for _ in range(N_NULL):
        perm = rng.integers(0, n, size=idx.shape)
        Aw = np.zeros((n, n))
        Aw[perm[:, 0], perm[:, 1]] = w_obs
        Aw = np.maximum(Aw, Aw.T)
        num_w = np.bincount(B.ravel(), weights=Aw.ravel(), minlength=B.max() + 1)
        f_w = np.divide(num_w, den, out=np.zeros_like(num_w), where=den > 0)
        q_w.append(q_of(Aw, mass * f_w[B], members))

    q_p = []
    for _ in range(N_NULL):
        perm = rng.permutation(len(members))
        D2 = haversine(lat[perm], lon[perm])
        B2 = bin_ids(D2)
        den2 = np.bincount(B2.ravel(), weights=mass.ravel(), minlength=B2.max() + 1)
        num2 = np.bincount(B2.ravel(), weights=A.ravel(), minlength=B2.max() + 1)
        f2 = np.divide(num2, den2, out=np.zeros_like(num2), where=den2 > 0)
        P2 = mass * f2[B2]
        q_p.append(q_of(A, P2, members))

    def z(q: float, qs: list[float]) -> float:
        s = np.std(qs)
        return (q - np.mean(qs)) / s if s > 1e-12 else 0.0

    return {"Q_Spa_obs": round(q_obs, 4), "Q_Spa_eshuffle_mean": round(float(np.mean(q_w)), 4),
            "Q_Spa_eshuffle_z": round(z(q_obs, q_w), 1),
            "Q_Spa_pshuffle_mean": round(float(np.mean(q_p)), 4),
            "Q_Spa_pshuffle_z": round(z(q_obs, q_p), 1),
            "Q_NG": round(q_ng(A, members), 4)}


def main() -> None:
    feats, ids, pairs, agg_by_id = load_ctx()
    rng = np.random.default_rng(SEED)
    lat, lon = coords(ids)
    D = haversine(lat, lon)
    B = bin_ids(D)
    N = population(ids)

    rows = []
    # ---- production: A = рёбра 12/2024 (веса = косинус) ----
    E = prod_edges()
    A = np.zeros((len(ids), len(ids)))
    np.add.at(A, (E[:, 0].astype(int), E[:, 1].astype(int)), E[:, 2])
    np.add.at(A, (E[:, 1].astype(int), E[:, 0].astype(int)), E[:, 2])
    A = A / 2  # симметричная (i,j)-пара учтена дважды в add.at
    m = prod_members(ids)
    r = evaluate(A, N, D, B, lat, lon, m, rng)
    r.update({"partition": "production", "ARI": round(ari_all(m, ids, agg_by_id), 3)})
    rows.append(r)

    # ---- typology14: A = полная матрица косинусов (>=0) ----
    X, cols, mt, q = typ14(feats, ids, pairs)
    At = np.clip(X @ X.T, 0.0, None)
    np.fill_diagonal(At, 0.0)
    r = evaluate(At, N, D, B, lat, lon, mt, rng)
    r.update({"partition": "typology14", "ARI": round(ari_all(mt, ids, agg_by_id), 3)})
    rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv("out/paper_exp1_gravity.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
