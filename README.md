# Выделяем экономические агломерации по данным от СберИндекса

Министерства выделяют официальные агломерации политически — в том числе из удобства: чтобы было понятно, куда и как направлять бюджеты. Но рынок не знает про административные границы и похожие люди живут по разные стороны муниципалитетов! Чтобы найти «похожих» соседей и сгруппировать их в _экономические_ агломерации мы посмотрели, сколько средств тратят пользователи Сбера и на какие категории товаров. Соединили муниципалитеты, связанные дорогами в пределах 50 км, и сгруппировали похожие по профилю расходов в рыночные кластеры.

Визуализация здесь: https://tsutsen.github.io/sber-agglomerations/

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
  clustering/       пайплайн кластеризации:
    run.py          финальный пайплайн (7 окон → Louvain → Jaccard)
    metrics.py      ICVI-метрики
    features.py     профиль потребления МО по месяцам
    config.py + config.yaml   все гиперпараметры
    network/graphs.py  дорога ≤ cutoff + топ-K по сходству → граф
    tune.py       сетка гиперпараметров → out/tune_grid.csv
    validate/     воспроизводимая валидация: нулевые модели, методы,
                  типология vs официальный перечень (3 скрипта, все работают)
    experiments/  все эксперименты из отчёта (19 скриптов)
methodology/        README.md (отчёт), EXPERIMENTS.md (журнал экспериментов)
visualization/      лендинг: index_basket.html (монолит), site/ (сайт),
                    landing/ (исходники), build_landing.py (сборка)
```

## Воспроизведение

Всё запускается из корня репозитория (Python 3.12+; окружение — `requirements.txt`;
для обновления визуализаций нужен node и `npm install` в `visualization/`):

```bash
python run_all.py all      # pipeline → metrics → validate → experiments → viz
python run_all.py experiments` пересобирает все эксперименты в `out/
python run_all.py --help   # описание pipeline, metrics, validate, experiments, viz, smoke

# или шаги по отдельности:
python -m scripts.clustering.run          # 7 окон → out/clusters_all.csv, out/jaccard.csv, out/compare_final.csv
python -m scripts.clustering.metrics      # ICVI-метрики → out/metrics.csv
python -m scripts.clustering.validate.official   # типология vs официальный перечень → out/cluster_vs_official.csv
```

Входные данные для финального пайплайна лежат в `data/sources/` 
Входные данные для экспериментов — в `data/experiments/inputs/`
