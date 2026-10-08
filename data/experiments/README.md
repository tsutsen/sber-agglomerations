# Эксперименты

Данные, использовавшиеся только в экспериментах, не вошедших в финал.
Финальные результаты (`../results/`) от этих файлов не зависят. Методика,
выводы и решения — `../../methodology/EXPERIMENTS.md`; скрипты —
`../../scripts/clustering/experiments/`; запуск — `python run_all.py experiments`.

```
inputs/
  raw/       исходные скачивания (xlsx/csv по городам и регионам, qol_cities/)
  derived/   таблицы, собранные из исходников по МО / официальным агломерациям
results/     зафиксированные результаты экспериментов (см. таблицу ниже)
```

## Входные данные (`inputs/`)

Всё, что читают скрипты экспериментов, кроме `../sources/` (таблицы пайплайна):

### `inputs/derived/`

| файл | что |
|---|---|
| `mo_names.csv` | `territory_id` → название МО (извлечено из `clusters.geojson`, который в репозиторий не входит — читался только словарь имён) |
| `mo_qol.csv` | QoL-индекс городов (2021–2024, 8 компонент) по МО — qol_stability, type_aggregates |
| `mo_static_features.csv` | статические оси признаков (зарплата, занятость, миграция, доступность рынков, сезонность, QoL) |
| `mo_mobility.csv` | мобильность по МО (СберИндекс) — design |
| `official_agglomerations_mapped.csv` | официальный перечень ВГП с маппингом на МО и регионами — ARI/purity в экспериментах; источник `../sources/official_agglomerations.csv` |
| `cluster_outlines.geojson` | объединённые контуры МО официальных агломераций — paper_exp2 (PP-компактность) |
| `cluster_names.csv` | имена типов отчитанной типологии — type_aggregates |
| `market_access_per_mo.csv`, `migration_per_mo.csv` | входные таблицы того же порядка, собранные по МО: `territory_id` + `market_access` / `value`; упоминаются в `EXPERIMENTS.md` (§2) |

### `inputs/raw/`

| файл | что |
|---|---|
| `qol_cities/` | погородские таблицы QoL (xlsx, по одному на город) — исходник `../derived/mo_qol.csv`. Единый дамп всех листов: `qol_all_cities.csv` (не коммитится; собрать: `pd.concat(pd.read_excel(f, sheet_name=None)[y] for f in glob("*.xlsx") for y in ...)`) |
| `sber_national_price_index_by_category.csv`, `sber_national_category_growth.csv` | всероссийские ценовой индекс и рост по категориям (Сбер) — дефляция G1–G3 |
| `sber_mobility_index.csv` | исходник `../derived/mo_mobility.csv` (СберИндекс) |

База расходов с геометрией — общая с пайплайном: таблица `../sources/derived/mo_consumption_population.csv`,
геометрия МО — `../results/geojson/mo.geojson` (2548 полигонов, ключ `territory_id`).
Дорожные расстояния — `../sources/matrix_distance_mo_pairs.csv` (3,3 млн пар,
совпадает с исходным `5_connection` Сбера построчно). Индекс корзины по регионам —
`../sources/basket_size_per_mo.csv`.

## Результаты экспериментов (воспроизведено 2025-10-08)

Таблицы в `results/` — выходные файлы соответствующих скриптов
(см. `EXPERIMENTS.md`, раздел каждого эксперимента):

| файл | эксперимент |
|---|---|
| `features_exp.csv`, `features_exp_extra.csv`, `features_exp_yoy.csv` | расширенное пространство признаков (A–D, окна, K15, YoY) |
| `features_exp3.csv`, `features_exp4.csv` | варианты сходства (S1–S5), ценовая дефляция и рост (G1–G3) |
| `jaccard.csv` | сходство составов кластеров между окнами |
| `paper_exp1_gravity.csv` | гравитационная модель (E1) |
| `paper_exp2_pp.csv`, `paper_exp2_pp_summary.csv` | PP-компактность (E2) |
| `paper_exp3_regionalization.csv` | региональная когерентность (E3) |
| `paper_exp4_cvi.csv` | CVI-стабильность (E4) |
| `paper_exp5_migration.csv`, `paper_exp5_migration_by_type.csv` | миграционные потоки (E5) |
| `paper_exp6_kmeans.csv` | kmeans-базлайн (E6) |
| `paper_exp7_gravity.csv` | гравитация на финальной типологии (E7) |
| `price_ineq.json`, `price_strat_dlevel.json` | неравенство цен внутри кластеров, стратификация по уровням цен |
| `qol_stability.json` | устойчивость QoL-профилей |
| `review2_ablations.csv` | абляции признаков (review 2) |
| `baselines.csv`, `baselines_pairwise_ARI.csv`, `methods_compare.csv` | базлайны и сравнение методов |
| `tune_grid.csv` | сетка подбора (knn, resolution) |
| `types_profile_g3.csv` | z-профили типов экономической типологии |
