# Архив: prep-скрипты исходного layout

Референсные копии вспомогательных скриптов (исходники — `src/...`
первоначального проекта). Они подготовили входные таблицы, которые сейчас
зафиксированы в `data/sources/` и `data/experiments/inputs/`, поэтому
пересобирать их не нужно: выход уже лежит в репозитории, а исходники
этих скриптов (загрузки с sberindex.ru) — нет. Пути в docstring'ах — от
layout исходного проекта.

| файл | что делал |
|---|---|
| `build_mo_dataset.py` | собрал `data/sources/derived/mo_consumption_population.{csv,gpkg}` и `mo_consumption_monthly_long.csv` из сырых паркетников Сбера + Росстата |
| `features_ext.py` | статические признаки МО → `mo_static_features.csv` (использовалось в экспериментах G1–G3) |
| `qol_download.py` → `qol_extract.py` → `qol_match.py` → `qol_to_mo.py` | цепочка: индекс качества жизни (развивай.рф) → `mo_qol.csv` → сопоставление с МО |
| `03_mobility.py` | «индекс мобильности» Сбера → `mo_mobility.csv` |
