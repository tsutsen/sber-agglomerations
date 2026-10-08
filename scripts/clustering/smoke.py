"""Дымовой проход: косинус-граф + Louvain на 100 крупнейших МО, сравнение с официальным перечнем."""
import igraph as ig
import numpy as np
import pandas as pd

DATA = "data/sources/derived"
CATS = ["food", "health", "cafe", "transport", "marketplace"]
D_CUTOFF = 100.0  # км, автодорожное расстояние
KNN = 25           # топ-k самых похожих соседей в пределах cutoff (мультиграф-окрестность)

# --- данные ------------------------------------------------------------------
dic = pd.read_csv(f"{DATA}/mo_consumption_population.csv")
mask = dic.has_consumption & dic.pop_2024.notna()
top = dic[mask]  # все МО с потреблением (топ-100 дают разреженный граф: города разнесены)
top_ids = top["territory_id"].to_numpy()
ids = set(top_ids)

m = pd.read_csv(f"{DATA}/mo_consumption_monthly_long.csv",
                usecols=["territory_id", "date", "cat", "consumption_rub"])
m = m[m.territory_id.isin(ids) & m.cat.isin(CATS)]
X = m.pivot_table(index="territory_id", columns=["date", "cat"], values="consumption_rub")
X = X.reindex(top.territory_id)  # строки: МО, колонки: (месяц, cat)
X = X.apply(pd.to_numeric, errors="coerce").fillna(X.mean())

# --- правило рёбер: средний cos сходства 5-категорных векторов по месяцам ----
v = X.to_numpy(dtype=float).reshape(len(ids), -1, len(CATS))  # (MO, month, cat)
norm = np.linalg.norm(v, axis=2)
cos = (v[:, None] * v[None]).sum(-1) / (norm[:, None, :] * norm[None, :, :]).clip(min=1e-9)
sim = cos.mean(axis=2)                                  # (MO, MO)
np.fill_diagonal(sim, 0)

# --- пространственный cutoff + топ-k по сходству ------------------------------
id2i = {t: i for i, t in enumerate(top_ids)}
# дорожная матрица из репозитория (OSM; см. data/sources/README.md)
conn = pd.read_csv("data/sources/matrix_distance_mo_pairs.csv")
conn = conn[conn.territory_id_x.isin(ids) & conn.territory_id_y.isin(ids) & (conn.distance <= D_CUTOFF)]
conn["i"] = conn.territory_id_x.map(id2i)
conn["j"] = conn.territory_id_y.map(id2i)
conn["w"] = sim[conn.i.to_numpy(), conn.j.to_numpy()]
kk = conn.sort_values("w", ascending=False).groupby("i").head(KNN)
pairs = {(min(a, b), max(a, b)) for a, b in zip(kk.i, kk.j, strict=True)}
edges = [(a, b, float(sim[a, b])) for a, b in sorted(pairs)]
print(f"kNN-рёбер (top{KNN} в пределах {D_CUTOFF} км): {len(edges)} (степень ~{2*len(edges)/len(ids):.0f})")

g = ig.Graph(len(ids), [(e[0], e[1]) for e in edges])
g.es["weight"] = [e[2] for e in edges]

# --- Louvain на kNN-графе -----------------------------------------------------
cl = g.community_multilevel(weights="weight")
member = np.asarray(cl.membership)
n_cl = int(member.max()) + 1
sizes = np.bincount(member)
print(f"кластеров: {n_cl} | модулярность: {cl.q:.3f} | топ-размеры: "
      f"{sorted(sizes.tolist(), reverse=True)[:15]}")

# --- сравнение с официальным перечнем ----------------------------------------
off = pd.read_csv("data/sources/official_agglomerations.csv").rename(
    columns={"agg_name": "Агломерация"})
off = off[off.territory_id.notna()]
off = off[off.territory_id.isin(ids)]
names = dict(zip(top_ids, top.municipal_district_name.to_numpy(), strict=True))
by_id = dict(zip(off.territory_id, off["Агломерация"], strict=True))
for cid in range(n_cl):
    members = top_ids[member == cid]
    if len(members) < 3:
        continue
    labels = sorted({by_id[t] for t in members if t in by_id})
    big = [names[t] for t in members][:8]
    print(f"\nкластер {cid} (n={len(members)}, офиц. агломерации: {list(labels) or '—'}):")
    print("  ", ", ".join(big), "…" if len(members) > 8 else "")
