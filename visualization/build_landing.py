#!/usr/bin/env python3
"""Собирает лендинг из visualization/landing/ (шаблон + css + js) и данных final_report.
Страницы: 1 — официальные vs рыночные агломерации, 2 — цены и зарплаты, 3 — методология.

Две сборки из одних же исходников:
  1) монолит visualization/index_basket.html — всё инлайном (file://, предпросмотр);
  2) веб-сборка visualization/site/ (index.html + app.js + data.json + geo.topojson) —
     для деплоя: кирпичи упрощены (mo_web.geojson, mapshaper 50%), геометрия — TopoJSON
     (общие границы один раз + квантизация ~1 м), грузится с gzip.
DATA: mo (МО), clusters (кластеры + тип vs официального), cores (ядра).
Геометрия: GEO — МО-кирпичи (атомы); clgs — union кирпичей кластера; official — union кирпичей
агломерации (геометрия строится из кирпичей, не берётся из готового geojson)."""
import json
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent          # final_report/
RES = ROOT / 'data' / 'results'
SRC = ROOT / 'data' / 'sources'
VIZ = ROOT / 'visualization'
LND = VIZ / 'landing'

# ---------- mo: id МО -> данные ----------
bf = pd.read_csv(RES / 'mo' / 'basket_final.csv')
mo = {}
for r in bf.itertuples(index=False):
    mo[int(r.territory_id)] = {
        'n': r.municipality_full, 'r': r.region,
        'fp': round(r.basket_fill, 1), 'fr': round(r.basket_fill_real, 1),
        's': int(r.salary), 'c': int(r.spatial_cluster),
    }

# ---------- clusters: id кластера -> {имя, размеры, индексы, тип vs официального} ----------
cs = pd.read_csv(RES / 'mo' / 'cluster_summary.csv')
cvo = pd.read_csv(RES / 'mo' / 'cluster_vs_official.csv')
# match/match_partial/match_expanded — одна категория на карте:
# «похожи на официальные агломерации, пересекаются с ними»
TY_MAP = {'match': 'match', 'match_partial': 'match', 'match_expanded': 'match',
          'merge': 'merge', 'new_market': 'new'}
dom = dict(zip(cvo.cluster, cvo.dominant_agg))
ty = dict(zip(cvo.cluster, cvo.type.map(TY_MAP)))
clusters = {int(r.cluster): {
    'n': r.cluster_name, 'k': int(r.n_mo),
    'fp': round(r.fill_price, 1), 'fr': round(r.fill_real, 1),
    's': int(r.mean_salary), 'ty': ty[int(r.cluster)],
    'dom': dom.get(int(r.cluster)),
} for r in cs.itertuples(index=False)}

# ---------- cores: id кластера -> [[название МО, формат активности, значение], ...] ----------
sc = pd.read_csv(RES / 'spatial_clusters.csv')
cores = {}
for r in sc[sc.core_rank.notna()].itertuples(index=False):
    cores.setdefault(int(r.cluster), []).append(
        [r.mo_name, r.activity_fmt, float(r.activity_rub_month)])
for cid in cores:
    cores[cid].sort(key=lambda x: x[2], reverse=True)

data = {'mo': {str(k): v for k, v in mo.items()},
        'clusters': {str(k): v for k, v in clusters.items()},
        'cores': {str(k): v for k, v in cores.items()}}


def clean(o):
    """NaN -> null: JSON.parse в браузере не принимает NaN (JS-литерал в монолите — да)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(x) for x in o]
    if isinstance(o, float) and o != o:
        return None
    return o


data = clean(data)

from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.strtree import STRtree


MIN_PART_AREA = 1e-4  # deg2 ≈ 0.8 км2: осколки от наложений сырых границ (усики);
# настоящий МО — всегда ≥ 3 км2, поэтому порог безопасен


def poly_parts(g):
    if g.geom_type == 'GeometryCollection':
        return [x for x in g.geoms if x.geom_type in ('Polygon', 'MultiPolygon') and x.area >= MIN_PART_AREA]
    return [g] if g.geom_type in ('Polygon', 'MultiPolygon') and g.area >= MIN_PART_AREA else []


def make_mo_clipped():
    """Стыкованные кирпичи из сырых mo.geojson: общий simplify(0.001) + последовательный
    клип (каждый МО вычитает union предыдущих, от больших к малым). Кэш — mo_clipped.geojson."""
    mgj = json.load(open(RES / 'geojson' / 'mo.geojson'))
    items = []
    for f in mgj['features']:
        g = shape(f['geometry'])
        if not g.is_valid:
            g = g.buffer(0)
        g = g.simplify(0.001, preserve_topology=True)  # ~100 м: общее для всех МО
        items.append((int(f['properties']['territory_id']), g, f['properties']))
    items.sort(key=lambda t: t[1].area, reverse=True)
    tree = STRtree([g for _, g, _ in items])
    out = []
    for i, (tid, g, p) in enumerate(items):
        nbs = [items[j][1] for j in tree.query(g, predicate='intersects') if j < i]
        g2 = g.difference(unary_union(nbs)) if nbs else g
        g2 = unary_union(poly_parts(g2)) if nbs else g
        if g2.is_empty:
            continue  # полностью анклав внутри чужого МО — на карте не виден
        out.append({'type': 'Feature', 'id': tid,
                    'properties': {'name': p['municipal_district_name'], 'region': p['region_name']},
                    'geometry': g2.__geo_interface__})
        items[i] = (tid, g2, p)  # дальше клипим уже по результату
    return out


def official_from_bricks(feats):
    """49 официальных агломераций. Состав — из CSV
    data/sources/official_agglomerations.csv (territory_id, agg_id, agg_name):
    добавить МО/агломерацию = правка строк CSV (имена МО не нужны — id уникальны).
    Геометрия — union тех же МО-кирпичей, что и кластеры (швы не разъезжаются).
    scattered пересчитывается из текущего spatial_cluster: МО агломерации
    разбросаны по >1 кластеру и доля доминирующего < 0.8 (правило compare/official.py)."""
    off = pd.read_csv(SRC / 'official_agglomerations.csv')
    by_tid = {f['id']: shape(f['geometry']) for f in feats}
    tid2cl = {t: d['c'] for t, d in mo.items()}
    out = {'type': 'FeatureCollection', 'features': []}
    for agg_id, g in off.groupby('agg_id', sort=True):
        tids = g.territory_id.tolist()
        labs = [tid2cl[t] for t in tids if t in tid2cl]
        vc = pd.Series(labs).value_counts()
        scattered = bool(len(vc) > 1 and vc.iloc[0] / max(len(labs), 1) < 0.8)
        names = [mo[t]['n'] for t in tids if t in mo]
        geoms = [by_tid[t] for t in tids if t in by_tid]
        u = clean_holes(unary_union(geoms))
        out['features'].append({'type': 'Feature', 'id': int(agg_id),
                                'properties': {'agg': g.agg_name.iloc[0], 'scattered': scattered,
                                               'mo_list': names},
                                'geometry': u.__geo_interface__})
    return out


def clean_holes(u):
    """Чистка union: выбрасывает части < MIN_PART_AREA (усиковые осколки) и
    ВЫРОЖДЕННЫЕ дырки (area < MIN_PART_AREA) — hairline-петли от последовательного
    клиппинга кирпичей. Реальные дырки (непокрытые МО, >= ~0.8 км2) сохраняются."""
    parts = []
    for p in (u.geoms if u.geom_type in ('MultiPolygon', 'GeometryCollection') else [u]):
        if p.geom_type not in ('Polygon', 'MultiPolygon'):
            continue
        for pp in (p.geoms if p.geom_type == 'MultiPolygon' else [p]):
            if pp.area < MIN_PART_AREA:
                continue
            holes = [h for h in pp.interiors if Polygon(h).area >= MIN_PART_AREA]
            parts.append(Polygon(pp.exterior, holes))
    if not parts:
        return Polygon()
    u2 = unary_union(parts)
    return u2


def clgs_from_bricks(feats):
    """Сплошные полигоны кластеров: union кирпичей кластера, без дырок и внутренних рёбер,
    без повторного упрощения (иначе швы разъедутся)."""
    by_cl = {}
    for f in feats:
        m = mo.get(f['id'])
        if m:
            by_cl.setdefault(m['c'], []).append(shape(f['geometry']))
    clgs = {}
    for cid, geoms in by_cl.items():
        polys = poly_parts(unary_union(geoms))
        if not polys:
            continue  # весь кластер — усиковые осколки, невидим
        if polys[0].geom_type == 'MultiPolygon':
            polys = polys[0].geoms
        solid = Polygon(polys[0].exterior) if len(polys) == 1 \
            else MultiPolygon([Polygon(p.exterior) for p in polys])
        clgs[str(cid)] = solid.__geo_interface__
    return clgs


# ---------- сборка 1: монолит (file://) ----------
CLIP = RES / 'geojson' / 'mo_clipped.geojson'
if CLIP.exists():
    feats = json.load(open(CLIP))['features']
    print(f'geo cache: {CLIP.name} ({len(feats)} feats)')
else:
    feats = make_mo_clipped()
    CLIP.write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}))
    print(f'geo cache written: {CLIP.name} ({len(feats)} feats)')
geo = {'type': 'FeatureCollection', 'features': feats}
data['official'] = official_from_bricks(feats)
data['clgs'] = clgs_from_bricks(feats)
# регионы РФ 2023 (фон подложки): оставляем фигуры с кодом num;
# из 91 в источнике попадают 86. «Республика Крым» и «Севастополь» в файле
# имеют num=0 (Крым разбит на две фигуры), поэтому дописаны явно
NO_NUM = {'Республика Крым', 'Севастополь'}
REGIONS = {'type': 'FeatureCollection',
           'features': [f for f in json.load(open(SRC / 'regions.geojson'))['features']
                        if (f.get('properties', {}).get('num')
                            or f['properties'].get('RUS_NAME') in NO_NUM)
                        and f.get('geometry') and f['geometry']['coordinates']]}
data['regions'] = REGIONS

tpl = open(LND / 'template.html', encoding='utf-8').read()
# preview-линк к style.css нужен только для прямого предпросмотра шаблона
tpl = re.sub(r'<link rel="stylesheet" href="style.css">.*?\n', '', tpl)
css = open(LND / 'style.css', encoding='utf-8').read()
JS_MONO = ['data.js', 'common.js', 'page1.js', 'page2.js', 'main.js']
js = '\n'.join(open(LND / 'js' / f, encoding='utf-8').read() for f in JS_MONO)
geo_js = json.dumps(geo, ensure_ascii=False, separators=(',', ':'))
data_js = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
tpl = (tpl
       .replace('/*__CSS__*/', css)
       .replace('/*__JS__*/', js)
       .replace('/*__DATA__*/null', data_js)
       .replace('/*__GEO__*/null', geo_js))
assert '__CSS__' not in tpl and '__JS__' not in tpl and '__DATA__' not in tpl and '__GEO__' not in tpl
out = VIZ / 'index_basket.html'
out.write_text(tpl, encoding='utf-8')
print(f'monolith: mo={len(mo)} clusters={len(clusters)} official={len(data["official"]["features"])} '
      f'feats={len(feats)} clgs={len(data["clgs"])}')
print(f'{out}  {out.stat().st_size / 1e6:.1f} MB')

# ---------- сборка 2: веб (site/, TopoJSON) ----------
feats_w = json.load(open(RES / 'geojson' / 'mo_web.geojson'))['features']
print(f'web bricks: mo_web.geojson ({len(feats_w)} feats)')
# Два МО (Коломенский, Павловский Посад — МО с вырезами) degenerирует
# mapshaper на 50% (коллапс вырезов — геометрия null): берём сырые
# контуры из mo.geojson. Швы могут расходиться с соседями на величину
# упрощения (~2.5 км) — не видно без глубокого зума.
if any(not f.get('geometry') for f in feats_w):
    raw = {int(f['properties']['territory_id']): f['geometry']
           for f in json.load(open(RES / 'geojson' / 'mo.geojson'))['features']}
    n_fix = 0
    for f in feats_w:
        if not f.get('geometry'):
            f['geometry'] = raw[f['id']]
            n_fix += 1
    print(f'web: {n_fix} degenerированных МО восстановлены из mo.geojson')
SITE = VIZ / 'site'
SITE.mkdir(exist_ok=True)
off_w = official_from_bricks(feats_w)
clgs_w = clgs_from_bricks(feats_w)
clgs_fc = {'type': 'FeatureCollection',
           'features': [{'type': 'Feature', 'id': int(c), 'geometry': g}
                        for c, g in clgs_w.items()]}
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / 'clgs.geojson').write_text(json.dumps(clgs_fc, ensure_ascii=False))
    (td / 'official.geojson').write_text(json.dumps(off_w, ensure_ascii=False))
    (td / 'regions.geojson').write_text(json.dumps(REGIONS, ensure_ascii=False))
    subprocess.run(['node', str(VIZ / 'topo_build.js'),
                    str(RES / 'geojson' / 'mo_web.geojson'),
                    str(td / 'clgs.geojson'), str(td / 'official.geojson'),
                    str(td / 'regions.geojson'),
                    str(SITE / 'geo.topojson')], check=True)
# official/clgs в data.json не нужны — они приходят из geo.topojson (bootstrap.js)
(SITE / 'data.json').write_text(
    json.dumps({k: v for k, v in data.items() if k not in ('official', 'clgs')},
               ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
JS_WEB = ['bootstrap.js', 'common.js', 'page1.js', 'page2.js', 'main.js']
(SITE / 'app.js').write_text(
    '\n'.join(open(LND / 'js' / f, encoding='utf-8').read() for f in JS_WEB), encoding='utf-8')
tpl_w = open(LND / 'template.html', encoding='utf-8').read()
tpl_w = re.sub(r'<link rel="stylesheet" href="style.css">.*?\n', '', tpl_w)
tpl_w = (tpl_w
         .replace('/*__CSS__*/', css)
         .replace('<script>/*__JS__*/</script>',
                  '<script src="https://cdn.jsdelivr.net/npm/topojson-client@3/dist/topojson-client.min.js"></'
                  'script>\n'
                  '<script type="module" src="app.js"></script>'))
assert '/*__JS__*/' not in tpl_w and '/*__DATA__*/' not in tpl_w
(SITE / 'index.html').write_text(tpl_w, encoding='utf-8')
tot = sum((SITE / f).stat().st_size for f in ('index.html', 'app.js', 'data.json', 'geo.topojson'))
print(f'web: official={len(off_w["features"])} clgs={len(clgs_w)} site total {tot / 1e6:.1f} MB raw')
