/* Веб-карта сервиса: область + период → POST /api/v1/query → термоточки, гари, справка, выгрузки. */
(function () {
  'use strict';

  const API = '/api/v1';
  const DEMO = { bbox: [39.25, 47.95, 39.50, 48.10], date_from: '2024-09-15', date_to: '2024-10-10' };
  const SEV_COLOR = { 1: '#f4d03f', 2: '#f39c12', 3: '#c0392b' };
  const SEV_LABEL = { 1: 'слабая', 2: 'средняя', 3: 'сильная' };
  const OUTLINE_COLOR = '#7a0c12';
  const NIGHT_STROKE = '#14306b';
  // шкала I4: 300 → 360 K, 5 ступеней (жёлтый → оранжевый → тёмно-красный)
  const I4_MIN = 300, I4_MAX = 360;
  const I4_COLORS = ['#ffe066', '#ffb347', '#f57c2a', '#d9381e', '#8b0f14'];
  const MAX_OUTLINE_SCARS = 3000;  // объединение полигонов в контур события — только для обозримых запросов
  const LANDCOVER = { 10: 'деревья', 20: 'кустарник', 30: 'трава', 40: 'пашня', 50: 'застройка', 60: 'голая почва',
    70: 'снег', 80: 'вода', 90: 'болото', 95: 'мангры', 100: 'мох' };
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 1) => Number(v).toLocaleString('ru-RU', { minimumFractionDigits: d, maximumFractionDigits: d });
  const fmtInt = (v) => Number(v).toLocaleString('ru-RU');
  const utc = (s) => (s || '').replace('T', ' ').replace('Z', '');

  function i4Color(t) {
    if (t == null || !Number.isFinite(t)) return '#bdbdbd';
    const k = Math.floor((t - I4_MIN) / ((I4_MAX - I4_MIN) / I4_COLORS.length));
    return I4_COLORS[Math.max(0, Math.min(I4_COLORS.length - 1, k))];
  }
  function hotRadius(conf, zoom) {  // 4–8 px по confidence (0.5 → 4, 1.0 → 8); нет значения → 5
    const r = conf == null || !Number.isFinite(conf) ? 5 : 4 + 4 * Math.max(0, Math.min(1, (conf - 0.5) / 0.5));
    // полный размер — с 10-го уровня; на обзорных масштабах тысячи точек иначе закрывают карту
    const k = zoom == null || zoom >= 10 ? 1 : Math.max(0.35, 1 - (10 - zoom) * 0.16);
    return Math.max(2, r * k);
  }

  // ── карта: canvas тянет десятки тысяч объектов. Гари, контур события и термоточки — на ОДНОМ canvas: верхний canvas
  //    перехватывает события мыши у нижних, и hover по гарям перестаёт работать; порядок внутри — bringToFront() ──
  const map = L.map('map', { preferCanvas: true }).setView([48.6, 43.5], 6);
  const pane = (name, z) => { map.createPane(name).style.zIndex = z; return L.canvas({ pane: name, padding: 0.3 }); };
  const rChips = pane('chips', 410), rData = pane('data', 420);
  const rScars = rData, rOutline = rData, rHot = rData;
  map.createPane('labels').style.zIndex = 405;
  map.getPane('labels').style.pointerEvents = 'none';
  map.getPane('chips').style.pointerEvents = 'none';

  // ── подложки ──
  const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/';
  const sat = L.layerGroup([
    L.tileLayer(ESRI + 'World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 18,
      attribution: 'Tiles © Esri — Esri, Maxar, Earthstar Geographics, and the GIS User Community' }),
    L.tileLayer(ESRI + 'Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}', { maxZoom: 18, pane: 'labels' }),
  ]);
  const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18, attribution: '© участники OpenStreetMap' });
  const BASEMAPS = { sat, osm, none: L.layerGroup() };
  let basemap = null, tileErrors = 0, tilesLoaded = 0, basemapNote = '', lastStatus = ['', false];
  function setBasemap(key, silent) {
    if (!silent) basemapNote = '';
    if (basemap) map.removeLayer(BASEMAPS[basemap]);
    basemap = key; tileErrors = 0; tilesLoaded = 0;
    BASEMAPS[key].addTo(map);
    document.body.classList.toggle('dark-map', key === 'sat');
    segSet('basemap', key);
    if (!silent) saveLocal('basemap', key);
  }
  function onTileError() {  // нет сети / сервис тайлов недоступен → «Нет»: остаются граница территории и охваты сцен
    if (++tileErrors >= 6 && tilesLoaded === 0 && basemap !== 'none') {
      const was = basemap === 'sat' ? 'Спутник' : 'Карта';
      setBasemap('none', true);
      $('ly-chips').checked = true; syncLayers();
      basemapNote = `Подложка «${was}» недоступна (нет сети) — включён режим «Нет»: граница территории и охваты сцен.`;
      setStatus(lastStatus[0], lastStatus[1]);
    }
  }
  [sat.getLayers()[0], osm].forEach((l) => { l.on('tileerror', onTileError); l.on('tileload', () => { tilesLoaded++; }); });

  // ── слои данных ──
  const aoiLayer = L.geoJSON(null, { interactive: false, renderer: rChips,
    style: (f) => f.id === 'aoi' ? { color: '#2f6fb0', weight: 2, fill: false }
                                  : { color: '#2f6fb0', weight: 1, dashArray: '4 4', fill: false, opacity: .7 } });
  const chipsLayer = L.geoJSON(null, { interactive: false, renderer: rChips,
    style: (f) => ({ color: f.properties.kind === 'bs' ? '#19a386' : '#9aa7b4', weight: f.properties.kind === 'bs' ? 1.2 : .6,
                     fill: false, opacity: .8 }) });
  let scarMode = 'severity';
  const scarStyle = (f) => {
    const c = SEV_COLOR[f.properties.severity_class] || '#888';
    return scarMode === 'severity' ? { color: c, weight: .8, opacity: 1, fillColor: c, fillOpacity: .45 }
                                   : { color: c, weight: 0, opacity: 0, fillColor: c, fillOpacity: 0 };
  };
  const scarsLayer = L.geoJSON(null, { renderer: rScars, style: scarStyle });
  const outlineLayer = L.geoJSON(null, { renderer: rOutline, interactive: false,
    style: { color: OUTLINE_COLOR, weight: 2.5, fill: false, opacity: 1 } });
  const hotLayer = L.geoJSON(null, {
    pointToLayer: (f, ll) => {
      const p = f.properties, night = p.daynight === 'night';
      return L.circleMarker(ll, { renderer: rHot, radius: hotRadius(p.confidence, map.getZoom()), fillColor: i4Color(p.i4_bt), fillOpacity: .95,
        color: night ? NIGHT_STROKE : '#ffffff', weight: night ? 2 : 1 });
    } });
  map.on('zoomend', () => { const z = map.getZoom(); hotLayer.eachLayer((l) => l.setRadius(hotRadius(l.feature.properties.confidence, z))); });
  const drawn = new L.FeatureGroup().addTo(map);
  L.control.scale({ imperial: false }).addTo(map);

  // ── подсказки и попапы ──
  const row = (k, v) => `<tr><td>${k}</td><td>${v == null || v === '' ? '—' : v}</td></tr>`;
  scarsLayer.on('mouseover', (e) => {
    if (scarMode !== 'severity') return;
    e.layer.setStyle({ fillOpacity: .7, weight: 1.4 });
  });
  scarsLayer.on('mouseout', (e) => scarsLayer.resetStyle(e.layer));
  scarsLayer.bindTooltip((l) => { const p = l.feature.properties; return `${p.severity_label} · ${fmt(p.area_ha, 2)} га`; },
    { sticky: true, className: 'tip' });
  scarsLayer.on('click', (e) => {
    const p = e.layer.feature.properties;
    L.popup().setLatLng(e.latlng).setContent(`<b class="t"><i class="sw" style="background:${SEV_COLOR[p.severity_class]}"></i>Гарь · ${p.severity_label} степень</b><table>` +
      row('ID контура', p.contour_id) + row('Класс', p.severity_class) + row('Площадь', fmt(p.area_ha, 2) + ' га') +
      row('Снимок до', p.date_pre) + row('Снимок после', p.date_post) + row('Чип', p.chip_id) + row('Пожар', p.fire_event_id) +
      row('Источник', p.source === 'gt' ? 'эталон' : 'модель') + '</table>').openOn(map);
  });
  hotLayer.bindTooltip((l) => {
    const p = l.feature.properties;
    return `<b>I4 ${fmt(p.i4_bt)} K</b> · ΔT ${fmt(p.i4_bt - p.i5_bt)} K<br>${utc(p.acq_datetime)} UTC<br>${p.satellite} · ${p.daynight === 'night' ? 'ночь' : 'день'}`;
  }, { sticky: true, className: 'tip' });
  hotLayer.on('click', (e) => {
    const p = e.layer.feature.properties;
    L.popup().setLatLng(e.latlng).setContent(`<b class="t"><i class="dot" style="background:${i4Color(p.i4_bt)}"></i>Термоточка VIIRS · ${p.daynight === 'night' ? 'ночь' : 'день'}</b><table>` +
      row('Время (UTC)', utc(p.acq_datetime)) + row('Спутник', p.satellite) +
      row('I4 (3.74 мкм)', fmt(p.i4_bt) + ' K') + row('I5 (11.45 мкм)', fmt(p.i5_bt) + ' K') + row('ΔT = I4 − I5', fmt(p.i4_bt - p.i5_bt) + ' K') +
      row('Покров', LANDCOVER[p.landcover] || p.landcover) +
      row('Уверенность', p.confidence == null ? null : fmt(p.confidence, 2)) +
      row('Чип', p.chip_id) + row('ID', p.hotspot_id) + row('Источник', p.source === 'gt' ? 'эталон' : 'модель') + '</table>').openOn(map);
  });

  // ── легенда на карте (внизу справа) ──
  const legend = L.control({ position: 'bottomright' });
  legend.onAdd = () => {
    const el = L.DomUtil.create('div', 'map-legend');
    const step = (I4_MAX - I4_MIN) / I4_COLORS.length;
    el.innerHTML =
      '<div class="lg-title">Термоточки · I4, K</div>' +
      `<div class="lg-grad">${I4_COLORS.map((c) => `<i style="background:${c}"></i>`).join('')}</div>` +
      `<div class="lg-ticks">${I4_COLORS.map((_, i) => `<span>${I4_MIN + i * step}</span>`).join('')}<span>${I4_MAX}</span></div>` +
      `<div class="lg-row"><i class="dot ring-day"></i>день <i class="dot ring-night"></i>ночь · размер — уверенность</div>` +
      '<div class="lg-title">Гари · степень поражения</div>' +
      `<div class="lg-row">${[1, 2, 3].map((k) => `<span><i class="sw" style="background:${SEV_COLOR[k]}"></i>${SEV_LABEL[k]}</span>`).join('')}</div>` +
      `<div class="lg-row"><i class="line" style="background:${OUTLINE_COLOR}"></i>контур события</div>`;
    L.DomEvent.disableClickPropagation(el);
    return el;
  };
  legend.addTo(map);

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
  const shapeStyle = { color: '#3b82f6', weight: 2, fillOpacity: .04, interactive: false, dashArray: '6 4' };
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
  function setPolygon(ring, fit) {  // ring: [[lon, lat], …] без замыкающей точки
    const poly = L.polygon(ring.map(([x, y]) => [y, x]), shapeStyle);
    setArea({ geometry: poly.toGeoJSON().geometry }, poly);
    if (fit) map.fitBounds(poly.getBounds(), { padding: [30, 30] });
  }
  const parseBbox = (s) => {
    const v = String(s).split(',').map(Number);
    return v.length === 4 && v.every(Number.isFinite) && v[0] < v[2] && v[1] < v[3] ? v : null;
  };
  $('bbox').addEventListener('change', () => {
    const v = parseBbox($('bbox').value);
    if (v) setBbox(v, true); else setStatus('bbox: нужно четыре числа minx,miny,maxx,maxy', true);
  });

  // ── мелкие элементы управления ──
  function segSet(id, value) { $(id).querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === value)); }
  function segBind(id, fn) { $(id).addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) fn(b.dataset.v); }); }
  function saveLocal(k, v) { try { localStorage.setItem('firemon.' + k, v); } catch (e) { /* приватный режим */ } }
  function loadLocal(k) { try { return localStorage.getItem('firemon.' + k); } catch (e) { return null; } }

  function syncLayers() {
    const want = [[hotLayer, $('ly-hot').checked], [scarsLayer, $('ly-scars').checked],
                  [outlineLayer, $('ly-scars').checked], [chipsLayer, $('ly-chips').checked], [aoiLayer, $('ly-aoi').checked]];
    want.forEach(([l, on]) => { if (on && !map.hasLayer(l)) l.addTo(map); if (!on && map.hasLayer(l)) map.removeLayer(l); });
    restack();
  }
  ['ly-hot', 'ly-scars', 'ly-chips', 'ly-aoi'].forEach((id) => $(id).addEventListener('change', syncLayers));
  segBind('basemap', (v) => setBasemap(v));
  segBind('scar-mode', (v) => { scarMode = v; segSet('scar-mode', v); applyScarMode(); });

  let outlineReady = false;
  function restack() {  // снизу вверх: гари → контур события → термоточки
    if (map.hasLayer(outlineLayer)) outlineLayer.bringToFront();
    if (map.hasLayer(hotLayer)) hotLayer.bringToFront();
  }
  function applyScarMode() {
    scarsLayer.setStyle(scarStyle);
    // «только контур» без посчитанного контура события (слишком большой запрос) → границы самих полигонов
    if (scarMode === 'outline' && !outlineReady) scarsLayer.setStyle({ color: OUTLINE_COLOR, weight: 1, opacity: 1, fillOpacity: 0 });
  }

  const source = () => ($('source-toggle').checked ? 'pred' : 'gt');
  function setSource(v) {
    $('source-toggle').checked = v === 'pred';
    $('src-gt').classList.toggle('on', v !== 'pred'); $('src-pred').classList.toggle('on', v === 'pred');
    $('source-info').textContent = v === 'pred' ? 'Показаны предсказания модулей AF и BS (на обучающих сценах).' : 'Показана эталонная разметка.';
  }
  $('source-toggle').addEventListener('change', () => { setSource(source()); if (area) runQuery(); });

  // ── контур события: объединение полигонов одного пожара (severity_class 0 в терминах справки) ──
  function buildOutline(features, token) {
    outlineLayer.clearLayers(); outlineReady = false;
    if (!features.length || typeof polygonClipping === 'undefined') return;
    if (features.length > MAX_OUTLINE_SCARS) { applyScarMode(); return; }
    const groups = new Map();
    features.forEach((f) => {
      const k = f.properties.fire_event_id || f.properties.chip_id;
      const g = f.geometry, polys = g.type === 'Polygon' ? [g.coordinates] : g.type === 'MultiPolygon' ? g.coordinates : [];
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(...polys);
    });
    const keys = [...groups.keys()];
    (function next(i) {
      if (token !== queryToken) return;  // пришёл новый запрос
      if (i >= keys.length) { outlineReady = true; applyScarMode(); restack(); return; }
      try {
        const polys = groups.get(keys[i]);
        const merged = polygonClipping.union(polys[0], ...polys.slice(1));
        outlineLayer.addData({ type: 'Feature', properties: { fire_event_id: keys[i], severity_class: 0 },
          geometry: { type: 'MultiPolygon', coordinates: merged } });
      } catch (e) { /* вырожденная геометрия — событие остаётся без общего контура */ }
      setTimeout(() => next(i + 1), 0);
    })(0);
  }

  // ── запрос ──
  function setStatus(text, isError) {
    lastStatus = [text || '', !!isError];
    $('status').textContent = [text, basemapNote].filter(Boolean).join(' ');
    $('status').classList.toggle('error', !!isError);
  }
  function setBusy(on) {
    $('btn-query').disabled = on; $('btn-demo').disabled = on;
    $('btn-query').classList.toggle('busy', on);
    $('btn-query').querySelector('.label').textContent = on ? 'Запрос…' : 'Запросить';
  }

  function writeUrl() {
    const q = new URLSearchParams();
    if (area && area.bbox) q.set('bbox', area.bbox.map((v) => +v.toFixed(5)).join(','));
    if (area && area.geometry) q.set('poly', area.geometry.coordinates[0].slice(0, -1).map((c) => `${+c[0].toFixed(5)},${+c[1].toFixed(5)}`).join(';'));
    if ($('date-from').value) q.set('from', $('date-from').value);
    if ($('date-to').value) q.set('to', $('date-to').value);
    q.set('source', source());
    history.replaceState(null, '', area ? '?' + q.toString() : location.pathname);
  }

  let queryToken = 0;
  async function runQuery() {
    if (!area) { setStatus('Сначала задайте область: нарисуйте её на карте, введите bbox или нажмите «Демо-запрос».', true); return; }
    const token = ++queryToken;
    const body = Object.assign({ source: source(), date_from: $('date-from').value || null, date_to: $('date-to').value || null }, area);
    setBusy(true); $('empty').hidden = true;
    setStatus('Выполняется запрос…');
    writeUrl();
    try {
      const t0 = performance.now();
      const resp = await fetch(API + '/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const data = await resp.json();
      if (!resp.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
      if (token !== queryToken) return;
      map.closePopup();
      scarsLayer.clearLayers(); hotLayer.clearLayers();
      scarsLayer.addData(data.burn_scars); hotLayer.addData(data.hotspots);
      buildOutline(data.burn_scars.features, token);
      applyScarMode(); restack();
      const n = data.meta, empty = n.n_hotspots + n.n_burn_scars === 0;
      $('cnt-hot').textContent = fmtInt(n.n_hotspots); $('cnt-scars').textContent = fmtInt(n.n_burn_scars);
      renderSummary(data.summary, data.query_id, data);
      $('summary').hidden = empty; $('empty').hidden = !empty;
      if (!empty) {  // к результату, а не к области: область бывает намного больше
        const b = L.latLngBounds([]);
        [scarsLayer, hotLayer].forEach((l) => { const lb = l.getBounds(); if (lb.isValid()) b.extend(lb); });
        if (b.isValid()) map.fitBounds(b, { padding: [40, 40], maxZoom: 14 });
      }
      setStatus(empty ? '' : `Найдено: термоточек ${fmtInt(n.n_hotspots)}, контуров гарей ${fmtInt(n.n_burn_scars)}. ` +
                `Сервер ${Math.round(n.elapsed_ms)} мс, всего ${Math.round(performance.now() - t0)} мс.` +
                (n.n_burn_scars > MAX_OUTLINE_SCARS ? ' Контур события строится при числе контуров до ' + fmtInt(MAX_OUTLINE_SCARS) + '.' : ''));
    } catch (err) {
      setStatus('Ошибка: ' + err.message, true);
    } finally {
      if (token === queryToken) setBusy(false);
    }
  }

  function renderSummary(s, qid, data) {
    $('sum-total').textContent = fmt(s.area_total_ha, 2);
    $('tiles').innerHTML = ['1', '2', '3'].map((k) => {
      const c = s.by_severity[k];
      return `<div class="tile" style="--c:${SEV_COLOR[k]}"><span class="name">${c.label}</span>` +
             `<span class="ha">${fmt(c.area_ha, 1)}<small> га</small></span><span class="pct">${fmt(c.share_pct)} %</span></div>`;
    }).join('');
    $('sharebar').innerHTML = ['1', '2', '3'].map((k) => `<i style="width:${s.by_severity[k].share_pct}%;background:${SEV_COLOR[k]}"></i>`).join('');
    const period = (s.period.date_from || 'начало архива') + ' — ' + (s.period.date_to || 'конец архива');
    const scenes = new Set();
    data.burn_scars.features.forEach((f) => scenes.add(f.properties.chip_id));
    data.hotspots.features.forEach((f) => scenes.add(f.properties.chip_id));
    $('summary-line').textContent = `термоточек ${fmtInt(s.n_hotspots)} · сцен ${fmtInt(scenes.size)} · ${period}`;
    $('summary-facts').innerHTML = [
      ['Источник масок', s.source === 'gt' ? 'эталон' : 'модель'],
      ['Контуров гарей', fmtInt(s.n_contours)], ['Пожаров (событий)', fmtInt(s.n_fire_events)],
      ['Термоточки: день / ночь', `${fmtInt(s.n_hotspots_day)} / ${fmtInt(s.n_hotspots_night)}`],
      ['Площадь области запроса', fmt(s.aoi_area_km2, 1) + ' км²'],
      ['Сформировано (UTC)', utc(s.generated_at)],
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

  function clearAll() {
    queryToken++;
    scarsLayer.clearLayers(); hotLayer.clearLayers(); outlineLayer.clearLayers(); outlineReady = false;
    map.closePopup(); setArea(null);
    $('summary').hidden = true; $('empty').hidden = true;
    $('cnt-hot').textContent = ''; $('cnt-scars').textContent = '';
    if (range) { $('date-from').value = range[0]; $('date-to').value = range[1]; }
    setStatus(''); setBusy(false);
    history.replaceState(null, '', location.pathname);
    if (extent) map.fitBounds([[extent[1], extent[0]], [extent[3], extent[2]]]);
  }

  function runDemo() {
    $('date-from').value = DEMO.date_from; $('date-to').value = DEMO.date_to;
    setBbox(DEMO.bbox, false);
    runQuery();
  }
  $('btn-query').addEventListener('click', runQuery);
  $('btn-demo').addEventListener('click', runDemo);
  $('btn-demo-2').addEventListener('click', runDemo);
  $('btn-clear').addEventListener('click', clearAll);

  // ── стартовое состояние: подложка, слои, граница, охваты, диапазон дат, параметры из URL ──
  const saved = loadLocal('basemap');
  setBasemap(saved && BASEMAPS[saved] ? saved : 'sat', true);
  setSource('gt');
  syncLayers();

  const params = new URLSearchParams(location.search);
  let extent = null, range = null;
  const hasUrlArea = params.has('bbox') || params.has('poly');
  fetch(API + '/aoi').then((r) => r.ok ? r.json() : null).then((g) => { if (g) aoiLayer.addData(g); }).catch(() => {});
  fetch(API + '/chips').then((r) => r.json()).then((g) => {
    chipsLayer.addData(g);
    $('cnt-chips').textContent = fmtInt(g.features.length);
    const b = chipsLayer.getBounds();
    if (b.isValid()) { extent = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]; if (!hasUrlArea) map.fitBounds(b); }
  }).catch(() => {});
  $('btn-extent').addEventListener('click', () => { if (extent) setBbox(extent, true); });

  function clampPreset(from, to) {
    if (!range) return;
    $('date-from').value = from < range[0] ? range[0] : from;
    $('date-to').value = to > range[1] ? range[1] : to;
    if (area) runQuery();
  }
  $('preset-2024').addEventListener('click', () => clampPreset('2024-01-01', '2024-12-31'));
  $('preset-all').addEventListener('click', () => { if (range) clampPreset(range[0], range[1]); });

  fetch(API + '/dates').then((r) => r.json()).then((d) => {
    const g = d.gt, mins = [g.hotspots.date_min, g.burn_scars.date_min].filter(Boolean).sort(),
      maxs = [g.hotspots.date_max, g.burn_scars.date_max].filter(Boolean).sort();
    if (!mins.length) { setStatus('Нет данных сервиса: запустите python scripts/build_service_data.py', true); return; }
    range = [mins[0], maxs[maxs.length - 1]];
    $('date-from').value = range[0]; $('date-to').value = range[1];  // по умолчанию — весь архив
    $('date-from').min = $('date-to').min = range[0]; $('date-from').max = $('date-to').max = range[1];
    const noPred = d.pred.hotspots.n + d.pred.burn_scars.n === 0;
    $('dates-info').textContent = `В архиве: ${range[0]} … ${range[1]}` + (noPred ? '. Слой «модель» не собран.' : '');
    $('source-toggle').disabled = noPred;

    // состояние из ссылки: область, даты, источник → запрос выполняется сразу
    if (params.get('from')) $('date-from').value = params.get('from');
    if (params.get('to')) $('date-to').value = params.get('to');
    if (params.get('source') === 'pred' && !noPred) setSource('pred');
    const bbox = params.has('bbox') ? parseBbox(params.get('bbox')) : null;
    const ring = params.has('poly') ? params.get('poly').split(';').map((p) => p.split(',').map(Number)) : null;
    if (bbox) { setBbox(bbox, true); runQuery(); }
    else if (ring && ring.length >= 3 && ring.every((c) => c.length === 2 && c.every(Number.isFinite))) { setPolygon(ring, true); runQuery(); }
  }).catch(() => {});
})();
