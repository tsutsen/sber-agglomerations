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
