# Конкурс СберИндекса: российские экономические агломерации

все исходные данные, все
результаты и все скрипты, из которых они получены. Каждый результат в
`data/results/` можно проследить до исходных файлов в `data/sources/` через
ссылку на скрипт, указанный в `data/results/README.md`.

## Что сделано

По 2144 муниципальным образованиям (МО) России построен слой
**рыночных кластеров**:

1. **Рыночные кластеры — агломерации по дорогам.** Соседство по дорожной
   доступности (радиус 50 км) + сходство профиля потребительских расходов
   (5 долей + уровень) → граф → кластеризация (Louvain), 7 скользящих окон
   2023–2024. Итог: 873 кластера, из них 134 плотные (≥3 МО). Валидация:
   ARI 0.72–0.74 и purity 0.95 по официальному перечню агломераций (Минэкономразвития),
   3637 рёбер (нулевые модели и полная валидация — в отчёте).

Над кластерами — **уровень цен и покупательcкая способность**

Полный отчёт: `methodology/README.md` (+ журнал экспериментов `methodology/EXPERIMENTS.md`). Интерактивный лендинг «Рыночные
агломерации» (3 страницы: агломерации / цены и зарплаты / методология) —
два формата: монолит `visualization/index_basket.html` (открывается
double-click'ом) и сайт `visualization/site/` (3.9 МБ, грузится за ~1 с
с gzip; открыть: `python3 -m http.server` из `site/`).

## Структура

```
run_all.py          единая точка входа (pipeline / metrics / validate / viz / smoke)
data/
  sources/          исходные данные (SberIndex, базовые таблицы)
  results/          зафиксированные финальные результаты: spatial_clusters.csv, metrics.csv, mo/, geojson/
  experiments/      результаты экспериментов (зафиксированные + входные в inputs/)
scripts/
  clustering/       самодостаточный пайплайн:
    run.py          финальный пайплайн (7 окон → Louvain → Jaccard)
    metrics.py      ICVI-метрики
    features.py     профиль потребления МО по месяцам
    config.py + config.yaml   все гиперпараметры (единственный источник)
    network/graphs.py  дорога ≤cutoff + топ-K по сходству → граф
    tune.py       сетка гиперпараметров → out/tune_grid.csv
    validate/     воспроизводимая валидация: нулевые модели, методы,
                  типология vs официальный перечень (3 скрипта, все работают)
    experiments/  все эксперименты из отчёта (19 скриптов, все работают);
                  вход — data/experiments/inputs/
  archive/          референсные копии prep-скриптов исходного layout
methodology/        README.md (отчёт), EXPERIMENTS.md (журнал экспериментов)
visualization/      лендинг: index_basket.html (монолит), site/ (сайт),
                    landing/ (исходники), build_landing.py (сборка)
```

## Воспроизведение

Всё запускается из корня репозитория (Python 3.12+; окружение — `requirements.txt`;
для `viz` — node и `npm install` в `visualization/`):

```bash
python run_all.py all      # pipeline → metrics → validate → experiments → viz
python run_all.py --help   # по шагам: pipeline, metrics, validate, experiments, viz, smoke

# или шаги напрямую:
python -m scripts.clustering.run          # 7 окон → out/clusters_all.csv, out/jaccard.csv, out/compare_final.csv
python -m scripts.clustering.metrics      # ICVI-метрики → out/metrics.csv
python -m scripts.clustering.validate.official   # типология vs официальный перечень → out/cluster_vs_official.csv
```

Вход — `data/sources/` (словарь — `data/sources/README.md`), вход
экспериментов — `data/experiments/inputs/`. Правило дерев результатов:
`out/` — что пайплайн написал в этом прогоне (в git не идёт);
`data/results/` и `data/experiments/` — зафиксированные результаты
отчитанного прогона, уже в репозитории. Свежий прогон воспроизводит их с
малым разбегом (Louvain без фиксированного seed; см. `methodology/EXPERIMENTS.md`),
методика и выводы — там же.

`python run_all.py experiments` пересобирает все эксперименты в `out/`
(порядок по зависимостям, ~10 минут).

Все гиперпараметры — в `scripts/clustering/config.yaml` (читается
`config.py`, используется пайплайном и валидацией).
