/* common.js — общее: данные, хелперы, навигация, фабрики карт и тултипов.
   Страницы: page1.js (официальные vs рыночные), page2.js (цены и зарплаты).
   Запуск: main.js. */

// ---------- данные ----------
const { mo: MO, clusters: CL, official: OFF, cores: CORES, regions: REG } = DATA;

// ---------- хелперы ----------
const $  = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
const rnd = Math.round;
const fmt = n => n.toLocaleString('ru');

// члены пространственного кластера: id кластера -> [id МО, ...]
const members = {};
for (const [id, m] of Object.entries(MO)) if (m.c >= 0) (members[m.c] ??= []).push(+id);

// сплошные полигоны пространственных кластеров (из DATA.clgs)
const CLGEO = { type: 'FeatureCollection',
  features: Object.entries(DATA.clgs).map(([c, g]) => ({ type: 'Feature', id: +c, geometry: g })) };

const GREY = '#cfd4dc', EDGE = '#344054';

// категории стр. 2: 0 — «цены хорошие» (>115), 1 — «нормальные» (85–115), 2 — «высокие» (<85)
const CATC = ['#6E75D0', '#31C896', '#E84A4A'];
const CAT_EDGE = ['#4A51A6', '#22916E', '#B23434'];
const CATN = ['Цены хорошие', 'Цены нормальные', 'Цены высокие'];
const catOf = v => v > 115 ? 0 : v >= 85 ? 1 : 2;

// ---------- навигация и страницы ----------
const inited = {}, maps = {};
$$('nav button').forEach(b => (b.onclick = () => show(b.dataset.p)));
$$('.chip[data-go]').forEach(b => (b.onclick = () => show(b.dataset.go)));

function show(p) {
  $$('nav button').forEach(b => b.classList.toggle('on', b.dataset.p == p));
  $$('.page').forEach(s => s.classList.toggle('on', s.id == 'p' + p));
  history.replaceState(null, '', '#' + p);
  if (p == 2 && !inited[2]) initMap2();
  for (const m of Object.values(maps)) setTimeout(() => m.invalidateSize(), 50);
}

// ---------- фабрики карт и слоёв ----------
function mkMap(id) {
  const m = L.map(id, { zoomControl: false, preferCanvas: true, zoomSnap: .25, attributionControl: false });
  L.control.zoom({ position: 'bottomright' }).addTo(m);
  maps[id] = m;
  return m;
}
const mkPoly = (style, each) => L.geoJson(GEO, { style, onEachFeature: each });

// ---------- тултипы ----------
const T = { sticky: true, direction: 'top', offset: [0, -16], className: 'tt', opacity: 1 };
const POP = { maxWidth: 340, minWidth: 250 };

// компактный тултип: заголовок, подпись, пары «ключ — значение», сноска
const tip = (t, c, meta, rows = [], hint) =>
  `<div class="t">${esc(t)}</div><div class="m"><i class="dot" style="background:${c}"></i>${meta}</div>` +
  rows.map(r => `<div class="r"><span>${r[0]}</span><b>${r[1]}</b></div>`).join('') +
  (hint ? `<div class="go">${hint}</div>` : '');

// шапка карточки/попопа: точка цвета рядом с названием, подзаголовок
const head = (t, c, s) =>
  `<div class="ph"><div><div class="pt">${esc(t)} <i class="dot" style="background:${c}"></i></div><div class="ps">${s}</div></div></div>`;

// МО вне выборки
const greyTip = f => tip(f.properties.name, GREY, esc(f.properties.region), [], 'вне выборки');

// тултип, закреплённый за кластером: сверху, либо снизу, если сверху не влезает
let cTip = null;
function showCTip(map, bounds, html) {
  if (cTip) cTip.remove();
  const ctr = bounds.getCenter();
  const up = map.latLngToContainerPoint([bounds.getNorth(), ctr.lng]).y > 140;
  cTip = L.tooltip({ direction: up ? 'top' : 'bottom', className: 'tt', opacity: 1 })
    .setLatLng([up ? bounds.getNorth() : bounds.getSouth(), ctr.lng])
    .setContent(html).addTo(map);
}
function hideCTip() { if (cTip) { cTip.remove(); cTip = null } }

// ---------- счётчики в легенде: считаются из данных, не хардкод ----------
const kc = { match: 0, merge: 0, new: 0, small: 0 }, cc = [0, 0, 0];
for (const c of Object.values(CL)) { if (c.k >= 3) kc[c.ty]++; else kc.small++; cc[catOf(c.fr)]++ }
const nOut = GEO.features.length - Object.keys(MO).length;
const cnt = (id, n) => { const el = document.getElementById(id); el?.closest('.row')?.querySelector('.cnt')?.replaceChildren(n) };
cnt('f_match', kc.match); cnt('f_merge', kc.merge); cnt('f_new', kc.new); cnt('f_small', kc.small);
cnt('f_off', OFF.features.length); cnt('f_out', nOut); cnt('f_reg', REG.features.length); cnt('f_reg2', REG.features.length);
cnt('f_cat0', cc[0]); cnt('f_cat1', cc[1]); cnt('f_cat2', cc[2]);
cnt('f_dense', kc.small); cnt('f_out2', nOut);

// ---------- сворачивание панелей ----------
$$('.panel').forEach(p => {
  p.querySelector('header').onclick = () => p.classList.toggle('collapsed');
  if (innerWidth < 720) p.classList.add('collapsed');
});
