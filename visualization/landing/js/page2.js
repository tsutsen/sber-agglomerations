/* page2.js — «Цены и зарплаты»: рыночные кластеры в трёх категориях по цене с учётом зарплаты. */

let L2, L2c, byId2c = {}, L2reg, hov2 = null, hovCat2 = null;

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
  const hot = hov2 === f.id, t = catOf(c.fr), dim = hovCat2 !== null && hovCat2 !== t;
  return { fillColor: CATC[t], fillOpacity: hot ? 1 : dim ? .1 : .8, color: CAT_EDGE[t], weight: hot ? 3 : 1, opacity: hot ? 1 : dim ? .08 : .7 };
};
const st2 = f => {
  const m = MO[f.id];
  if (m && m.c >= 0) {
    const on = $('#f_mo2').checked && catOn(m.c) && (hovCat2 === null || catOf(CL[m.c].fr) === hovCat2);
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

// подсветка по наведению на строку легенды (как на стр.1)
[0, 1, 2].forEach(t => {
  const row = $('#f_cat' + t).closest('.row');
  row.addEventListener('mouseenter', () => { hovCat2 = t; L2c && L2c.setStyle(st2c); L2 && L2.setStyle(st2) });
  row.addEventListener('mouseleave', () => { hovCat2 = null; L2c && L2c.setStyle(st2c); L2 && L2.setStyle(st2) });
});

const P2_ALL = ['f_cat0', 'f_cat1', 'f_cat2', 'f_out2', 'f_dense', 'f_reg2'];
const p2SetAll = v => { P2_ALL.forEach(i => $('#' + i).checked = v); if (L2) { L2.setStyle(st2); L2c.setStyle(st2c); L2reg.setStyle(st2reg) } };
$('#p2_on').onclick = () => p2SetAll(true);
$('#p2_off').onclick = () => p2SetAll(false);
