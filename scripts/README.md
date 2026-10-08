# Скрипты

Единая точка входа — `run_all.py` в корне (см. корневой README, раздел
«Воспроизведение»). Каждый шаг — отдельный модуль, запускаемый и напрямую:
`python -m scripts.clustering.run` и т.п. (из корня репозитория).

## clustering/ — самодостаточный пайплайн

Всё, что нужно финальному пайплайну, — в этой папке; внешних зависимостей
от остального дерева нет.

| файл | что делает |
|---|---|
| `run.py` | **финальный пайплайн**: признаки → графы (7 окон) → Louvain → `out/clusters_all.csv`, `out/edges_*.csv`, `out/jaccard.csv`, `out/compare_final.csv` |
| `metrics.py` | ICVI-метрики (SW, CH, S_Dbw, AVI, AVU, MQ) → `out/metrics.csv` |
| `tune.py` | сетка knn × res по чистоте → `out/tune_grid.csv` |
| `smoke.py` | быстрый дымовой тест на подмножестве МО |
| `features.py` | профиль потребления МО по месяцам (5 долей + log уровня) из `data/sources/derived/` |
| `config.py` + `config.yaml` | **все гиперпараметры** (единственный источник; читаются `network/graphs.py`) |
| `network/graphs.py` | `candidate_pairs` (дорога ≤50 км), `window_edges` (топ-K по cos), `louvain` |

### validate/ — воспроизводимая валидация (все три запускаются на данных репозитория)

| файл | что делает |
|---|---|
| `baselines.py` | **нулевые модели**: distance_top8, distance_only, attribute_only, random → `out/baselines*.csv` |
| `methods.py` | сравнение сходств/алгоритмов (cos/pearson × louvain/agglomerative) → `out/methods_compare.csv` |
| `official.py` | типология кластеров vs официальный перечень → `out/cluster_vs_official.csv`, `out/official_coverage.csv` |

### experiments/ — все эксперименты из отчёта (все работают)

Вход — `data/experiments/inputs/`, выход — `out/`. Запуск целиком:
`python run_all.py experiments` (порядок по зависимостям: `features_exp4`
→ `basket_index` → … → `basket_final` → `type_aggregates`; ~10 минут).
Результаты отчитанного прогона зафиксированы в `data/experiments/`,
методика и выводы — `methodology/EXPERIMENTS.md`.

| файл | серия |
|---|---|
| `features_exp.py` | варианты A (12 осей), B (без расстояний), C, D (YoY) |
| `features_exp2.py` | E1–E4, гибриды, K15, w12/w24; типология B12 |
| `features_exp3.py` | S1–S5: детерминированные варианты сходства |
| `features_exp4.py` | G1–G3: ценовая дефляция, избыточный рост, типология 14 осей |
| `papers_common.py` | общая инфраструктура экспериментов (coherence, null, permutation) |
| `paper_exp1_gravity.py` | гравитационная модель: edge/position shuffle |
| `paper_exp2_pp.py` | компактность Polsby–Popper против null-наборов |
| `paper_exp3_regionalization.py` | Ward + connectivity, SKATER vs mutual-kNN + Louvain |
| `paper_exp4_cvi.py` | ICVI-метрики против null-модели |
| `paper_exp5_migration.py` | миграция vs расстояние до крупного города, vs типы |
| `paper_exp6_kmeans.py` | расширенный k-means в дуальном пространстве |
| `paper_exp7_gravity.py` | гравитационное ре-ранжирование рёбер |
| `review2.py` | ablations, run-to-run, k-чувствительность, holdout по ФО |
| `price_strat.py` | стратификация ценовой премии по уровням расходов |
| `price_ineq.py` | ценовой премиум: декомпозиция, тест пар, динамика |
| `qol_stability.py` | QoL-стабильность типов (отрицательная проверка) |
| `basket_cluster.py` | кластеризация по fill_real — отрицательный результат (градиент, не структура) |
| `basket_index.py` | **уровень цен и покупательная способность**: basket_fill, fill_real, hours_for_1000 |
| `basket_final.py` | **сборка `basket_final.csv`** (финальная таблица МО) |
| `design.py` | диагностика структуры графа (блоки, расстояния, мобильность) |

## archive/ — референсные копии prep-скриптов исходного layout

Скрипты подготовки данных и устаревшая визуализация из исходного проекта
(`src/`): `build_mo_dataset.py` (собирал `data/sources/derived/` из сырых
паркетников Сбера), цепочка `qol_*`, `03_mobility.py`, `features_ext.py`,
три viz-скрипта (суперсeded `visualization/build_landing.py`). Входные
файлы — вне репозитория; пути в docstring'ах — от layout исходного проекта.
