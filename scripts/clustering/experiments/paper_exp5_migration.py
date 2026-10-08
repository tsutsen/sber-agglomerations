"""EXP 5: миграционная валидация (Karachurina & Mkrtchyan, 2019).

Предсказание статьи: периферия теряет население (чистая миграция < 0),
центры — притягивают; интенсивность миграции зависит от расстояния до
крупного центра. Проверяем на наших данных (mig_per_1000 из статических
признаков):
  (1) Спирмен: mig_per_1000 против расстояния до ближайшего крупного
      города (>= 300 тыс. + Москва и СПб как единые точки);
  (2) медиана миграции по децилям расстояния (монотонность градиента);
  (3) медиана миграции по типам типологии 14 (типы «региональный центр»
      с осью mig_per_1000 должны лидировать, «периферия» — замыкать).

Эксперимент methodology/EXPERIMENTS.md: входные — data/experiments/inputs/, результаты — data/experiments/results/.paper_exp5_migration.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .papers_common import (coords, haversine, load_ctx, population,
                                   static14, typ14)


def main() -> None:
    feats, ids, pairs, _ = load_ctx()
    lat, lon = coords(ids)
    N = population(ids)
    static = static14(ids)
    mig = static.mig_per_1000.to_numpy(float)

    pop = pd.read_csv("data/sources/derived/mo_consumption_population.csv",
                      usecols=["territory_id", "region_name", "pop_2024"])
    p = pop.set_index("territory_id").reindex(ids)

    # якоря: МО >= 300 тыс. + Москва и СПб (как единые точки по центроидам МО)
    big_mask = (p.pop_2024.fillna(0) >= 300_000).to_numpy()
    anch_pos = np.where(big_mask)[0]
    city_w: dict[str, tuple] = {}
    for city in ("Москва", "Санкт-Петербург"):
        m = (p.region_name == city).to_numpy()
        city_w[city] = (lat[m], lon[m])

    def haversine_pts(lat_a, lon_a, lat_b, lon_b) -> np.ndarray:
        R = 6371.0
        la1, lo1 = np.deg2rad(lat_a)[:, None], np.deg2rad(lon_a)[:, None]
        la2, lo2 = np.deg2rad(lat_b)[None, :], np.deg2rad(lon_b)[None, :]
        h = (np.sin((la2 - la1) / 2) ** 2
             + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
        return 2 * R * np.arcsin(np.clip(np.sqrt(h), 0, 1))

    d_big = haversine_pts(lat, lon, lat[anch_pos], lon[anch_pos])
    d = d_big.min(axis=1)
    for city, (cla, clo) in city_w.items():
        dd = haversine_pts(lat, lon, cla, clo)
        d = np.minimum(d, dd.min(axis=1))
    # самокрупный город: расстояние до самого себя = 0 — корректно (он и есть центр)

    df = pd.DataFrame({
        "mo": ids,
        "mig_per_1000": mig,
        "dist_to_big_city_km": d,
        "population": N,
    })
    X, cols, mt, q = typ14(feats, ids, pairs)
    df["type"] = mt
    df.to_csv("out/paper_exp5_migration.csv", index=False)

    rho, pval = spearmanr(df.mig_per_1000, df.dist_to_big_city_km)
    print(f"\nСпирмен mig vs dist: rho={rho:.3f}, p={pval:.3g}")

    dec = df.groupby(pd.qcut(df.dist_to_big_city_km, 10, labels=False,
                             duplicates="drop"))["mig_per_1000"].agg(
        ["median", "mean", "size"])
    print("\n=== по децилям расстояния до крупного города ===")
    print(dec.round(3).to_string())

    near = df[df.dist_to_big_city_km <= 50].mig_per_1000.median()
    far = df[df.dist_to_big_city_km >= 500].mig_per_1000.median()
    print(f"\nмедиана mig <=50 км: {near:.2f} | >=500 км: {far:.2f}")

    tp = df.groupby("type")["mig_per_1000"].agg(["median", "size"]).round(3)
    tp = tp.sort_values("median", ascending=False)
    tp.to_csv("out/paper_exp5_migration_by_type.csv")
    print("\n=== по типам типологии 14 ===")
    print(tp.to_string())


if __name__ == "__main__":
    main()
