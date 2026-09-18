/* Веб-карта сервиса: область + период → POST /api/v1/query → термоточки, гари, справка, выгрузки. */
(function () {
  'use strict';

  const API = '/api/v1';
  const DEMO = { bbox: [39.25, 47.95, 39.50, 48.10], date_from: '2024-09-15', date_to: '2024-10-10' };
  const SEV_COLOR = { 0: '#555', 1: '#f4d03f', 2: '#f39c12', 3: '#c0392b' };
  const LANDCOVER = { 10: 'деревья', 20: 'кустарник', 30: 'трава', 40: 'пашня', 50: 'застройка', 60: 'голая почва',
    70: 'снег', 80: 'вода', 90: 'болото', 95: 'мангры', 100: 'мох' };
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 1) => Number(v).toLocaleString('ru-RU', { minimumFractionDigits: d, maximumFractionDigits: d });

  // ── карта: canvas-рендерер тянет десятки тысяч объектов ──
  const map = L.map('map', { preferCanvas: true, renderer: L.canvas({ padding: 0.3 }) }).setView([48.6, 43.5], 6);
  const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    { maxZoom: 18, attribution: '© участники OpenStreetMap' });
  const blank = L.layerGroup();  // «пустая» подложка: без сети остаются граница AOI и охваты сцен
  osm.addTo(map);
  let tileErrors = 0;
  osm.on('tileerror', () => {
    if (++tileErrors === 8 && map.hasLayer(osm)) {
      map.removeLayer(osm); blank.addTo(map);
      setStatus('Подложка OSM недоступна (нет сети) — включена пустая подложка с границей территории.');
    }
  });

  const aoiLayer = L.geoJSON(null, { interactive: false,
    style: (f) => f.id === 'aoi' ? { color: '#1f4e79', weight: 2, fill: false }
                                  : { color: '#1f4e79', weight: 1, dashArray: '4 4', fill: false, opacity: .6 } }).addTo(map);
  const chipsLayer = L.geoJSON(null, { interactive: false,
    style: (f) => ({ color: f.properties.kind === 'bs' ? '#0e7c66' : '#7a8794', weight: f.properties.kind === 'bs' ? 1.2 : .6,
                     fill: false, opacity: .7 }) });
  const scarsLayer = L.geoJSON(null, {
    style: (f) => ({ color: SEV_COLOR[f.properties.severity_class], weight: .6, fillColor: SEV_COLOR[f.properties.severity_class],
                     fillOpacity: .65 }) }).addTo(map);
  const hotLayer = L.geoJSON(null, {
    pointToLayer: (f, ll) => L.circleMarker(ll, { radius: 4, weight: 1, color: '#fff', fillOpacity: .9,
      fillColor: f.properties.daynight === 'night' ? '#8e44ad' : '#e74c3c' }) }).addTo(map);
  const drawn = new L.FeatureGroup().addTo(map);

  L.control.layers({ 'OpenStreetMap': osm, 'Без подложки': blank },
    { 'Граница территории': aoiLayer, 'Охваты сцен (AF — серые, BS — зелёные)': chipsLayer, 'Гари': scarsLayer,
      'Термоточки': hotLayer, 'Область запроса': drawn }, { collapsed: true }).addTo(map);
  L.control.scale({ imperial: false }).addTo(map);

  const row = (k, v) => `<tr><td>${k}</td><td>${v == null || v === '' ? '—' : v}</td></tr>`;
  scarsLayer.on('click', (e) => {
    const p = e.layer.feature.properties;
    L.popup().setLatLng(e.latlng).setContent(`<b class="t">Гарь · степень: ${p.severity_label}</b><table>` +
      row('Площадь', fmt(p.area_ha, 2) + ' га') + row('Класс', p.severity_class) + row('ID контура', p.contour_id) +
      row('Пожар', p.fire_event_id) + row('Снимок до', p.date_pre) + row('Снимок после', p.date_post) +
      row('Источник', p.source) + '</table>').openOn(map);
  });
  hotLayer.on('click', (e) => {
    const p = e.layer.feature.properties;
    L.popup().setLatLng(e.latlng).setContent(`<b class="t">Термоточка VIIRS · ${p.daynight === 'night' ? 'ночь' : 'день'}</b><table>` +
      row('Время (UTC)', p.acq_datetime.replace('T', ' ').replace('Z', '')) + row('Спутник', p.satellite) +
      row('I4 (3.74 мкм)', fmt(p.i4_bt) + ' K') + row('I5 (11.45 мкм)', fmt(p.i5_bt) + ' K') +
      row('Покров', LANDCOVER[p.landcover] || p.landcover) +
      row('Уверенность', p.confidence == null ? null : fmt(p.confidence, 2)) +
      row('ID', p.hotspot_id) + row('Источник', p.source) + '</table>').openOn(map);
  });

  // ── рисование области ──
  Object.assign(L.drawLocal.draw.toolbar.buttons, { polygon: 'Нарисовать полигон', rectangle: 'Нарисовать прямоугольник' });
  Object.assign(L.drawLocal.draw.toolbar.actions, { text: 'Отмена', title: 'Отменить рисование' });
  Object.assign(L.drawLocal.draw.toolbar.finish, { text: 'Готово', title: 'Завершить' });
  Object.assign(L.drawLocal.draw.toolbar.undo, { text: 'Удалить точку', title: 'Удалить последнюю точку' });
  L.drawLocal.draw.handlers.rectangle.tooltip.start = 'Нажмите и тяните, чтобы нарисовать прямоугольник';
  L.drawLocal.draw.handlers.simpleshape.tooltip.end = 'Отпустите кнопку мыши';
  Object.assign(L.drawLocal.draw.handlers.polygon.tooltip, { start: 'Кликните, чтобы начать полигон',
    cont: 'Кликните, чтобы продолжить', end: 'Кликните по первой точке, чтобы замкнуть' });
  Object.assign(L.drawLocal.edit.toolbar.buttons, { edit: 'Изменить область', editDisabled: 'Нет области',
    remove: 'Удалить область', removeDisabled: 'Нет области' });
  Object.assign(L.drawLocal.edit.toolbar.actions.save, { text: 'Сохранить', title: 'Сохранить' });
  Object.assign(L.drawLocal.edit.toolbar.actions.cancel, { text: 'Отмена', title: 'Отменить' });
  Object.assign(L.drawLocal.edit.toolbar.actions.clearAll, { text: 'Удалить всё', title: 'Удалить всё' });
  const shapeStyle = { color: '#1565c0', weight: 2, fillOpacity: .04, interactive: false };
  map.addControl(new L.Control.Draw({
    draw: { polyline: false, marker: false, circle: false, circlemarker: false,
            rectangle: { shapeOptions: shapeStyle }, polygon: { shapeOptions: shapeStyle, allowIntersection: false } },
    edit: { featureGroup: drawn } }));

  let area = null;  // {bbox:[…]} | {geometry:{…}}
  function setArea(a, layer) {
    area = a;
    drawn.clearLayers();
    if (layer) drawn.addLayer(layer);
    $('bbox').value = a && a.bbox ? a.bbox.map((v) => v.toFixed(4)).join(',') : '';
    $('area-info').textContent = a && a.geometry ? 'Область: нарисованный полигон (' + (a.geometry.coordinates[0].length - 1) + ' вершин)' : '';
  }
  function areaFromLayer(layer) {
    if (layer instanceof L.Rectangle) {
      const b = layer.getBounds();
      return { bbox: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()] };
    }
    return { geometry: layer.toGeoJSON().geometry };
  }
  map.on(L.Draw.Event.CREATED, (e) => setArea(areaFromLayer(e.layer), e.layer));
  map.on(L.Draw.Event.EDITED, () => drawn.eachLayer((l) => setArea(areaFromLayer(l), l)));
  map.on(L.Draw.Event.DELETED, () => setArea(null));

  function setBbox(bbox, fit) {
    const rect = L.rectangle([[bbox[1], bbox[0]], [bbox[3], bbox[2]]], shapeStyle);
    setArea({ bbox }, rect);
    if (fit) map.fitBounds(rect.getBounds(), { padding: [30, 30] });
  }
  $('bbox').addEventListener('change', () => {
    const v = $('bbox').value.split(',').map(Number);
    if (v.length === 4 && v.every(Number.isFinite) && v[0] < v[2] && v[1] < v[3]) setBbox(v, true);
    else setStatus('bbox: нужно четыре числа minx,miny,maxx,maxy', true);
  });

  // ── запрос ──
  function setStatus(text, isError) {
    $('status').textContent = text || '';
    $('status').classList.toggle('error', !!isError);
  }
  const source = () => document.querySelector('input[name=source]:checked').value;

  async function runQuery() {
    if (!area) { setStatus('Сначала задайте область: нарисуйте её на карте, введите bbox или нажмите «Демо-запрос».', true); return; }
    const body = Object.assign({ source: source(), date_from: $('date-from').value || null, date_to: $('date-to').value || null }, area);
    $('btn-query').disabled = true;
    setStatus('Выполняется запрос…');
    try {
      const t0 = performance.now();
      const resp = await fetch(API + '/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const data = await resp.json();
      if (!resp.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
      scarsLayer.clearLayers(); hotLayer.clearLayers();
      scarsLayer.addData(data.burn_scars); hotLayer.addData(data.hotspots);
      hotLayer.bringToFront();
      renderSummary(data.summary, data.query_id);
      const n = data.meta;
      setStatus(`Найдено: термоточек ${n.n_hotspots}, контуров гарей ${n.n_burn_scars}. ` +
                `Сервер ${Math.round(n.elapsed_ms)} мс, всего ${Math.round(performance.now() - t0)} мс.` +
                (n.n_hotspots + n.n_burn_scars === 0 ? ' В этой области за период данных нет — попробуйте «Демо-запрос».' : ''));
    } catch (err) {
      setStatus('Ошибка: ' + err.message, true);
    } finally {
      $('btn-query').disabled = false;
    }
  }

  function renderSummary(s, qid) {
    $('summary').hidden = false;
    $('summary-rows').innerHTML = ['1', '2', '3'].map((k) => {
      const c = s.by_severity[k];
      return `<tr><td><i class="sw" style="background:${SEV_COLOR[k]}"></i>${k} — ${c.label}</td><td>${fmt(c.area_ha, 2)}</td><td>${fmt(c.share_pct)}</td></tr>`;
    }).join('');
    $('sum-total').textContent = fmt(s.area_total_ha, 2);
    $('sharebar').innerHTML = ['1', '2', '3'].map((k) => `<i style="width:${s.by_severity[k].share_pct}%;background:${SEV_COLOR[k]}"></i>`).join('');
    const period = (s.period.date_from || 'начало архива') + ' — ' + (s.period.date_to || 'конец архива');
    $('summary-facts').innerHTML = [
      ['Период', period], ['Источник масок', s.source === 'gt' ? 'эталон' : 'модель'],
      ['Контуров гарей', s.n_contours], ['Пожаров (событий)', s.n_fire_events],
      ['Термоточек', `${s.n_hotspots} (день ${s.n_hotspots_day} / ночь ${s.n_hotspots_night})`],
      ['Площадь области запроса', fmt(s.aoi_area_km2, 1) + ' км²'],
      ['Сформировано (UTC)', s.generated_at.replace('T', ' ').replace('Z', '')],
    ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
    $('crs-note').textContent = s.crs_note;
    const q = `${API}/query/${qid}`;
    $('dl-scars-geojson').href = `${q}/burn_scars?format=geojson`;
    $('dl-scars-shp').href = `${q}/burn_scars?format=shp`;
    $('dl-hot-geojson').href = `${q}/hotspots?format=geojson`;
    $('dl-hot-shp').href = `${q}/hotspots?format=shp`;
    $('dl-sum-json').href = `${q}/summary?format=json`;
    $('dl-sum-csv').href = `${q}/summary?format=csv`;
  }

  $('btn-query').addEventListener('click', runQuery);
  $('btn-demo').addEventListener('click', () => {
    $('date-from').value = DEMO.date_from; $('date-to').value = DEMO.date_to;
    setBbox(DEMO.bbox, true);
    runQuery();
  });

  // ── стартовые данные: граница AOI, охваты сцен, диапазон дат ──
  let extent = null;
  fetch(API + '/aoi').then((r) => r.ok ? r.json() : null).then((g) => { if (g) aoiLayer.addData(g); }).catch(() => {});
  fetch(API + '/chips').then((r) => r.json()).then((g) => {
    chipsLayer.addData(g);
    const b = chipsLayer.getBounds();
    if (b.isValid()) { extent = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]; map.fitBounds(b); }
  }).catch(() => {});
  $('btn-extent').addEventListener('click', () => { if (extent) setBbox(extent, true); });
  fetch(API + '/dates').then((r) => r.json()).then((d) => {
    const g = d.gt, mins = [g.hotspots.date_min, g.burn_scars.date_min].filter(Boolean).sort(),
      maxs = [g.hotspots.date_max, g.burn_scars.date_max].filter(Boolean).sort();
    if (!mins.length) { setStatus('Нет данных сервиса: запустите python scripts/build_service_data.py', true); return; }
    $('date-from').value = mins[0]; $('date-to').value = maxs[maxs.length - 1];
    $('dates-info').textContent = `В архиве: термоточки ${g.hotspots.date_min} … ${g.hotspots.date_max}, гари ${g.burn_scars.date_min} … ${g.burn_scars.date_max}` +
      (d.pred.hotspots.n + d.pred.burn_scars.n === 0 ? '. Слой «модель» не собран.' : '');
  }).catch(() => {});
})();
