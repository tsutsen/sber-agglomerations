# Результаты

Только финальные таблицы кластеризации и визуализации. Всё, что не вошло
в финал (нулевые модели, валидация, эксперименты), в репозиторий не включено.

## Файлы

| файл | содержимое |
|---|---|
| `spatial_clusters.csv` | **Итоговая кластеризация** (финальное окно 2024-12): 873 кластера, имена (топ-3 МО по плотности), состав (cluster, cluster_name, core_rank, territory_id, mo_name, activity) |
| `metrics.csv` | Метрики финальной кластеризации (metric, value): silhouette, CH, S_Dbw, AVI/AVU (сравнение с официальным перечнем), modularity |
| `mo/basket_final.csv` | **Главная таблица**: 2144 МО × {municipality, region, spatial_cluster, cores, salary, salary_index, hours_for_1000, price_index_2024, basket_fill, basket_fill_real} |
| `mo/cluster_summary.csv` | 873 spatial-кластера финального окна: id, n МО, ядра, уровень цен / покупательская способность, средняя зарплата — сводка для карты (стр. 2) |
| `mo/cluster_vs_official.csv` | 873 кластера: `type` (match/match_partial/match_expanded/merge/new_market), `dominant_agg`, `n_mo` — сравнение с официальным перечнем (раскраска стр. 1 карты) |
| `geojson/mo.geojson` | 2548 МО, только границы (без раскраски): territory_id, название, region. Вход сборки: из него строятся «кирпичи» (мозаика без швов) |
| `geojson/mo_web.geojson` | То же, упрощено (mapshaper 50%) — вход в TopoJSON для сайта |

## Метрики кластеризации

Метрики качества кластеризации (SW, CH, S_Dbw, AVI, AVU, MQ) и сравнение
с официальным перечнем — `cluster_vs_official.csv` (133 плотных кластеров,
≥3 МО, против 49 официальных агломераций). Методология и валидация —
`../methodology/README.md` и `../methodology/EXPERIMENTS.md`.

## Определения (таблицы МО)

- `basket_fill = 100 / price_index_2024` — доля стандартной корзины за 1000 руб.
- `basket_fill_real = basket_fill × salary_index / 100` — доля корзины на среднюю месячную зарплату (зарплата МО делится на среднюю зарплату по выборке МО).
- `hours_for_1000 = 1000 / salary × (173/1000)` — часов среднего труда на 1000 руб. (173 = 21.75 дн × 8 ч/день).

Важно: цены — регионального уровня (внутри региона все МО имеют одинаковый
price_index_2024); внутри spatial-кластера basket_fill однороден,
basket_fill_real — нет (CV 10–16%).
