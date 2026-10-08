/* bootstrap.js — загрузка данных для ВЕБ-сборки (site/): data.json + geo.topojson.
   Монолит (index_basket.html, file://) использует data.js с инлайнованными DATA/GEO.
   TopoJSON декодируется topojson-client (CDN в шаблоне): общие границы хранятся
   одним арком, поэтому швы МО/кластеров/агломераций совпадают по построению. */
const _d = await (await fetch('data.json')).json();
const _topo = await (await fetch('geo.topojson')).json();
const _cl = topojson.feature(_topo, _topo.objects.clusters);
const DATA = Object.assign(_d, {
  official: topojson.feature(_topo, _topo.objects.official),
  regions: topojson.feature(_topo, _topo.objects.regions),
  clgs: Object.fromEntries(_cl.features.map(f => [String(f.id), f.geometry])),
});
const GEO = topojson.feature(_topo, _topo.objects.mo);
