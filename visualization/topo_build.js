/* topo_build.js — сборка TopoJSON для веб-версии.
   Общие границы кирпичей/кластеров/агломераций хранятся ОДИН раз (арк-дедуп +
   квантизация 1e5 ~ 1 м), поэтому стыки не разъезжаются по построению.
   Использование: node topo_build.js <mo.geojson> <clgs.geojson> <official.geojson> <out.topojson> */
const fs = require('fs');
const topojson = require('topojson-server');
const [mo, clgs, off, regions, out] = process.argv.slice(2);
const topo = topojson.topology({
  mo: JSON.parse(fs.readFileSync(mo, 'utf8')),
  clusters: JSON.parse(fs.readFileSync(clgs, 'utf8')),
  official: JSON.parse(fs.readFileSync(off, 'utf8')),
  regions: JSON.parse(fs.readFileSync(regions, 'utf8')),
}, 1e5);
fs.writeFileSync(out, JSON.stringify(topo));
console.log('topo:', out, (fs.statSync(out).size / 1e6).toFixed(1), 'MB');
