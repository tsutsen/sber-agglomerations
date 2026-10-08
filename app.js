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

/* page1.js — «Официальные vs рыночные»: два слоя (базовая заливка кластеров + МО)
   и оверлей официальных агломераций. */

const TYPE_COLOR = { match: '#6E75D0', merge: '#DD8FB7', new: '#E84A4A', small: '#D9D9D9' };
const TYPE_EDGE = { match: '#4A51A6', merge: '#A86285', new: '#B23434', small: '#A6A6A6' };
const TYPE_RU = { match: 'Похожи на официальные агломерации, пересекаются с ними', merge: 'Объединяют сразу несколько официальных агломераций', new: 'Новые рыночные аггломерации, не пересекаются с официальными', small: 'Малые (1–2 МО)' };
const FKEY = { match: 'f_match', merge: 'f_merge', new: 'f_new', small: 'f_small' };

let L1, L1c, byId1c = {}, L1off, L1reg, hov1 = null, hovKey1 = null;

// ключ кластера: тип (для 3+ МО) или small
const keyOf = c => c.k >= 3 ? c.ty : 'small';
const keyOn = k => $('#' + FKEY[k]).checked;
const colOf = k => TYPE_COLOR[k];

// базовая заливка кластера
const st1c = f => {
  const c = CL[f.id];
  if (!c) return { fillColor: '#fff', fillOpacity: 0, weight: 0, opacity: 0, color: EDGE };
  const k = keyOf(c), on = keyOn(k);
  if (!on) return { fillColor: '#fff', fillOpacity: 0, weight: 0, opacity: 0, color: EDGE };
  const hot = hov1 === f.id, dim = hovKey1 !== null && hovKey1 !== k;
  return {
    fillColor: colOf(k), fillOpacity: hot ? 1 : dim ? .1 : .85,
    color: TYPE_EDGE[k], weight: hot ? 3 : 1.2, opacity: hot ? 1 : dim ? .08 : .7,
  };
};

// слой МО: тонкие границы (f_mo) + серые вне выборки (f_out)
const st1 = f => {
  const m = MO[f.id];
  if (m && m.c >= 0) {
    const on = $('#f_mo').checked && keyOn(keyOf(CL[m.c]));
    return { fillOpacity: 0, color: EDGE, weight: on ? .45 : 0, opacity: on ? .45 : 0 };
  }
  return { fillColor: '#fff', fillOpacity: $('#f_out').checked ? .5 : 0, color: '#000', weight: .8, opacity: $('#f_out').checked ? .3 : 0 };
};

// подложка «регионы» (фоновая, интерактива нет)
const st1reg = () => {
  const on = $('#f_reg').checked;
  return { fillColor: '#eceff3', fillOpacity: on ? 1 : 0, color: '#6b7480', weight: 1.2, opacity: on ? .9 : 0 };
};

// оверлей официальных агломераций: чёрный пунктир + полупрозрачная жёлтая заливка
const OFF_C = '#D3C52B';
const offVis = f => $('#f_off').checked;
const st1o = f => {
  if (!offVis(f)) return { fillOpacity: 0, opacity: 0, weight: 0, color: '#fff' };
  return { fillColor: '#000', fillOpacity: .12, color: '#000', weight: 2.5, opacity: .9, dashArray: '3 2' };
};

// список регионов, где стоят МО кластера (в порядке вхождения; обычно 1)
function regionOf(id) {
  const out = [];
  for (const i of members[id]) if (!out.includes(MO[i].r)) out.push(MO[i].r);
  return out.join(', ');
}

function popup1(id) {
  const c = CL[id];
  if (!keyOn(keyOf(c))) return '';
  const k = keyOf(c), rg = regionOf(id), cores = (CORES[id] || []);
  const mx = cores.length ? Math.max(...cores.map(x => x[2])) : 0;
  const moList = (members[id] || []).map(i => MO[i].n);
  return head(c.n, colOf(k), (rg ? rg + ' · ' : '') + c.k + ' МО · ' + TYPE_RU[k])
    + `<div class="chips"><span class="chip">${c.k} МО</span><span class="chip">зарплата ${fmt(c.s)} ₽/мес</span>${c.dom ? `<span class="chip">агломерация: ${esc(c.dom)}</span>` : ''}</div>`
    + (cores.length ? `<div class="sec">Ядра по экономической активности</div><div class="bars">${cores.map(x =>
      `<span>${esc(x[0])}</span><span class="v">${esc(x[1])}</span><div class="bar"><i style="width:${mx ? x[2] / mx * 100 : 0}%"></i></div>`).join('')}</div>` : '')
    + `<details><summary style="color:var(--accent)">Состав (${moList.length} МО)</summary><div class="mol">${moList.map(esc).join('<br>')}</div></details>`;
}

function initMap1() {
  const m = mkMap('map1');
  m.setView([60, 95], 3.25);
  // подложка: регионы РФ 2023
  L1reg = L.geoJson(REG, { style: st1reg, interactive: false }).addTo(m);
  L1c = L.geoJson(CLGEO, { style: st1c, interactive: false }).addTo(m);
  L1c.eachLayer(l => byId1c[l.feature.id] = l);
  L1 = mkPoly(st1, (f, l) => {
    const d = MO[f.id], nm = d ? d.n : f.properties.name, rg = d ? d.r : f.properties.region;
    l.on('mouseover', () => {
      if (!d || !keyOn(keyOf(CL[d.c]))) return;
      hov1 = d.c;
      L1c.setStyle(st1c);
      l.setStyle({ fillOpacity: 0, weight: 1.8, color: '#101828', opacity: 1 });
      l.bringToFront();
      const c0 = CL[d.c], k0 = keyOf(c0), r0 = regionOf(d.c);
      showCTip(m, byId1c[d.c].getBounds(), tip(c0.n, colOf(k0), `${c0.k} МО · ${esc(d.n)}`,
        [['Тип', TYPE_RU[k0]], ['Средняя зарплата', fmt(c0.s) + ' ₽/мес'], ...(r0 ? [[(r0.includes(',') ? 'Регионы' : 'Регион'), esc(r0)]] : [])],
        'клик — состав кластера'));
    });
    l.on('mouseout', () => { hov1 = null; L1c.setStyle(st1c); l.setStyle(st1(f)); hideCTip() });
    if (d && d.c >= 0) l.bindPopup(() => popup1(d.c), POP);
    else l.bindTooltip(greyTip({ properties: { name: nm, region: rg } }), T);
  }).addTo(m);
  L1off = L.geoJson(OFF, {
    style: st1o,
    onEachFeature: (f, l) => {
      l.on('mouseover', () => l.setStyle({ ...st1o(f), weight: 3.5 }));
      l.on('mouseout', () => l.setStyle(st1o(f)));
      l.bindTooltip(tip(f.properties.agg, OFF_C, 'официальная агломерация', [], 'клик — состав'), T);
      l.bindPopup(() => offVis(f) ? head(f.properties.agg, OFF_C, 'официальная агломерация')
        + `<div class="sec">Состав (${f.properties.mo_list.length} МО)</div><div class="mol">${f.properties.mo_list.map(esc).join('<br>')}</div>` : '', POP);
    },
  }).addTo(m);
}

// легенда: чекбокс — видимость; наведение на строку — подсветка на карте
['f_match', 'f_merge', 'f_new', 'f_small'].forEach(i => $('#' + i).onchange = () => {
  L1c && (L1c.setStyle(st1c), L1.setStyle(st1));
});
['f_out', 'f_mo'].forEach(i => $('#' + i).onchange = () => L1 && L1.setStyle(st1));
$('#f_reg').onchange = () => L1reg && L1reg.setStyle(st1reg);
['f_off'].forEach(i => $('#' + i).onchange = () => L1off && L1off.setStyle(st1o));

// подсветка по наведению на строку легенды
const rowKey = { f_match: 'match', f_merge: 'merge', f_new: 'new', f_small: 'small' };
Object.entries(rowKey).forEach(([i, k]) => {
  const row = $('#' + i).closest('.row');
  row.addEventListener('mouseenter', () => { hovKey1 = k; L1c && L1c.setStyle(st1c) });
  row.addEventListener('mouseleave', () => { hovKey1 = null; L1c && L1c.setStyle(st1c) });
});

// все / ничего
const P1_ALL = ['f_match', 'f_merge', 'f_new', 'f_small', 'f_off', 'f_reg'];
const p1SetAll = v => { P1_ALL.forEach(i => $('#' + i).checked = v); L1c && (L1c.setStyle(st1c), L1.setStyle(st1), L1off.setStyle(st1o)) };
$('#p1_on').onclick = () => p1SetAll(true);
$('#p1_off').onclick = () => p1SetAll(false);

/* page2.js — «Цены и зарплаты»: рыночные кластеры в трёх категориях по цене с учётом зарплаты. */

let L2, L2c, byId2c = {}, L2reg, hov2 = null;

// включён ли кластер: по категории + одиночные/парные (f_dense)
const catOn = id => {
  const c = CL[id];
  if (!c) return true;
  if (!$('#f_dense').checked && c.k < 3) return false;
  return $('#f_cat' + catOf(c.fr)).checked;
};

const st2c = f => {
  const c = CL[f.id];
  if (!c || !catOn(f.id)) return { fillColor: '#fff', fillOpacity: 0, weight: 0, opacity: 0, color: EDGE };
  const hot = hov2 === f.id, t = catOf(c.fr);
  return { fillColor: CATC[t], fillOpacity: hot ? 1 : .8, color: CAT_EDGE[t], weight: hot ? 3 : 1, opacity: hot ? 1 : .7 };
};
const st2 = f => {
  const m = MO[f.id];
  if (m && m.c >= 0) {
    const on = $('#f_mo2').checked && catOn(m.c);
    return { fillOpacity: 0, color: EDGE, weight: on ? .45 : 0, opacity: on ? .45 : 0 };
  }
  return { fillColor: '#fff', fillOpacity: $('#f_out2').checked ? .5 : 0, color: '#000', weight: .8, opacity: $('#f_out2').checked ? .3 : 0 };
};

// подложка «регионы» на стр.2 (свой чекбокс f_reg2)
const st2reg = () => {
  const on = $('#f_reg2').checked;
  return { fillColor: '#eceff3', fillOpacity: on ? 1 : 0, color: '#6b7480', weight: 1.2, opacity: on ? .9 : 0 };
};


// список регионов, где стоят МО кластера (в порядке вхождения; обычно 1)
const regionOf2 = id => {
  const out = [];
  for (const i of (members[id] || [])) if (!out.includes(MO[i].r)) out.push(MO[i].r);
  return out.join(', ');
};

function initMap2() {
  inited[2] = 1;
  const m = mkMap('map2');
  m.setView([60, 95], 3.25);
  // подложка: регионы РФ 2023 (та же, что на стр.1)
  L2reg = L.geoJson(REG, { style: st2reg, interactive: false }).addTo(m);
  L2c = L.geoJson(CLGEO, { style: st2c, interactive: false }).addTo(m);
  L2c.eachLayer(l => byId2c[l.feature.id] = l);
  L2 = mkPoly(st2, (f, l) => {
    const d = MO[f.id], nm = d ? d.n : f.properties.name, rg = d ? d.r : f.properties.region;
    l.on('mouseover', () => {
      if (!d || !catOn(d.c)) return;
      hov2 = d.c;
      L2c.setStyle(st2c);
      l.setStyle({ fillOpacity: 0, weight: 1.8, color: '#101828', opacity: 1 });
      l.bringToFront();
      const c0 = CL[d.c], t = catOf(c0.fr), r0 = regionOf2(d.c);
      showCTip(m, byId2c[d.c].getBounds(), tip(c0.n, CATC[t], `${c0.k} МО · ${esc(d.n)}`,
        [['Категория', CATN[t]], ['Уровень цен', c0.fp + '%'], ['Покупат. способность', c0.fr + '%'], ['Средняя зарплата', fmt(c0.s) + ' ₽/мес'], ...(r0 ? [[(r0.includes(',') ? 'Регионы' : 'Регион'), esc(r0)]] : [])],
        'клик — состав кластера'));
    });
    l.on('mouseout', () => { hov2 = null; L2c.setStyle(st2c); l.setStyle(st2(f)); hideCTip() });
    if (d && d.c >= 0) l.bindPopup(() => {
      if (!catOn(d.c)) return '';
      const c = CL[d.c], t = catOf(c.fr), r0 = regionOf2(d.c);
      const ms = (members[d.c] || []).map(i => MO[i]).sort((a, b) => b.fr - a.fr), mx = Math.max(...ms.map(x => x.fr), 100);
      return head(c.n, CATC[t], (r0 ? r0 + ' · ' : '') + c.k + ' МО · ' + CATN[t])
        + `<div class="chips"><span class="chip" title="100 = среднее по РФ">уровень цен ${c.fp}%</span><span class="chip" title="100 = среднее по РФ: корзина на среднюю зарплату">покупательская способность ${c.fr}%</span><span class="chip">${fmt(c.s)} ₽/мес</span></div>`
        + `<div class="sec">Покупательская способность по МО</div><div class="bars">${ms.slice(0, 6).map(x =>
          `<span>${esc(x.n)}</span><span class="v">${x.fr}%</span><div class="bar"><i style="width:${x.fr / mx * 100}%"></i></div>`).join('')}</div>`
        + `<details><summary style="color:var(--accent)">Состав (${ms.length} МО)</summary><div class="mol">${ms.map(x => esc(x.n)).join('<br>')}</div></details>`;
    }, POP);
    else l.bindTooltip(greyTip({ properties: { name: nm, region: rg } }), T);
  }).addTo(m);
}

['f_out2', 'f_mo2', 'f_dense', 'f_cat0', 'f_cat1', 'f_cat2', 'f_reg2']
  .forEach(i => $('#' + i).onchange = () => { if (L2) { L2.setStyle(st2); L2c.setStyle(st2c); L2reg.setStyle(st2reg) } });

const P2_ALL = ['f_cat0', 'f_cat1', 'f_cat2', 'f_out2', 'f_dense', 'f_reg2'];
const p2SetAll = v => { P2_ALL.forEach(i => $('#' + i).checked = v); if (L2) { L2.setStyle(st2); L2c.setStyle(st2c); L2reg.setStyle(st2reg) } };
$('#p2_on').onclick = () => p2SetAll(true);
$('#p2_off').onclick = () => p2SetAll(false);

/* main.js — запуск: карта 1-й страницы, стартовая страница. */
initMap1();
show(+location.hash.slice(1) || 1);
