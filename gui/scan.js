// Core Explorer: "Scan a log" landing page and "Scanned logs" library.
// Loaded after index.html's main script and uses its globals (LOGS, byId, wells, WELL_LOC, decode, openLog, setView, ...).
// Scanning needs the local scanner (python3 server.py); browsing scans already made works from file:// too.

// =============================================================== SCAN: upload + mark three boxes
const TOOLS = [
  ["depth_column", "Depth column", "#1f6feb", "The narrow column of depth numbers (9,100 · 9,200 …). Drag from the first number down to the last."],
  ["interpreted_lithology_column", "Rock column", "#d9480f", "The column of rock-pattern symbols (often headed “Lithology”). Use the same top and bottom as the depth column."],
  ["lithology_legend", "Legend", "#2f9e44", "The key that names each rock symbol, often titled “Lithology symbols” or “Rock types”."],
];
const SC = {info: null, page: 0, boxes: {}, tool: TOOLS[0][0], status: null, busy: false};

async function checkScanner() {
  const el = $("#scan-status"), drop = $("#drop");
  $("#n-ds").textContent = DS.length;
  if (!SERVER) {
    drop.classList.add("off"); $("#file").disabled = true;
    el.innerHTML = `<div class="note warn"><b>The scanner isn't running.</b> This page was opened as a file, so it can show logs but can't scan new ones.
      In Terminal, go to the <code>gui</code> folder and run <code>python3 server.py</code>. It opens this page at <code>http://localhost:8765</code>, ready to scan.</div>`;
    return;
  }
  try {
    const s = SC.status = await (await fetch("api/status")).json();
    if (s.missing.length) {
      el.innerHTML = `<div class="note bad"><b>The scanner needs a few things installed first:</b><ul style="margin:6px 0 0;padding-left:18px">${s.missing.map(m => `<li>${esc(m)}</li>`).join("")}</ul>
        Install them, then restart <code>python3 server.py</code>.</div>`;
    } else el.innerHTML = `<div class="note ok">✓ Scanner ready.</div>`;
  } catch (e) {
    el.innerHTML = `<div class="note bad">Couldn't reach the scanner. Is <code>python3 server.py</code> still running?</div>`;
  }
}
function showScan() {
  if (!SC.info) { $("#scan-start").hidden = false; $("#mark").hidden = true; }
  if (!SC.checked) { SC.checked = true; checkScanner(); }
}

// file picking / dropping
const drop = $("#drop");
$("#file").onchange = e => { pickFiles(e.target.files); $("#file").value = ""; };
["dragenter", "dragover"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => { if (SERVER) pickFiles(e.dataTransfer.files); });

// The scan and its bounding-box JSON can come together (pick or drop both), or the JSON on its own after the scan.
const isJson = f => /\.json$/i.test(f.name);
async function pickFiles(list) {
  const files = [...(list || [])], scan = files.find(f => !isJson(f)), json = files.find(isJson);
  const el = $("#scan-status");
  if (!scan) {
    if (json) el.innerHTML = `<div class="note bad">That's only the bounding-box JSON. Choose the log image (.tif) too: select or drop both files together.</div>`;
    return;
  }
  let bbox = null;
  if (json) {
    try { bbox = await readBoxJson(json); }
    catch (e) { el.innerHTML = `<div class="note bad">${esc(e.message)}</div>`; return; }
  }
  uploadFile(scan, bbox);
}
async function readBoxJson(f) {
  let j;
  try { j = JSON.parse(await f.text()); } catch (e) { throw new Error(`${f.name} isn't valid JSON.`); }
  const bb = j.bounding_boxes || j;
  const missing = TOOLS.map(t => t[0]).filter(k => !(Array.isArray(bb[k]) && bb[k].length === 4 && bb[k].every(Number.isFinite)));
  if (missing.length) throw new Error(`${f.name} has no ${missing.map(k => k.replace(/_/g, " ")).join(", ")} box. Expected "bounding_boxes" with depth_column, interpreted_lithology_column and lithology_legend as [x0, y0, x1, y1].`);
  return {name: f.name, width: +j.width || 0, height: +j.height || 0, boxes: Object.fromEntries(TOOLS.map(([k]) => [k, bb[k].map(Number)]))};
}
// JSON boxes are in the original image's pixels; the PDF page may render at another size, so scale to it.
function applyBoxJson(bbox) {
  const p = pg(), sx = bbox.width ? p.w / bbox.width : 1, sy = bbox.height ? p.h / bbox.height : 1;
  SC.boxes = {};
  for (const [k, [x0, y0, x1, y1]] of Object.entries(bbox.boxes)) {
    const b = [Math.min(x0, x1) * sx, Math.min(y0, y1) * sy, Math.max(x0, x1) * sx, Math.max(y0, y1) * sy]
      .map((v, i) => Math.round(Math.max(0, Math.min(i % 2 ? p.h : p.w, v))));
    if (b[2] - b[0] >= 4 && b[3] - b[1] >= 4) SC.boxes[k] = b;
  }
  const n = Object.keys(SC.boxes).length, next = TOOLS.find(([k]) => !SC.boxes[k]);
  SC.tool = next ? next[0] : TOOLS[0][0];
  const scaled = Math.abs(sx - 1) > 0.01 || Math.abs(sy - 1) > 0.01;
  $("#mk-msg").innerHTML = n === 3
    ? `<div class="note ok">✓ Boxes loaded from <b>${esc(bbox.name)}</b>${scaled ? ` (scaled from ${fmt(bbox.width)} × ${fmt(bbox.height)} px to this page)` : ""}. Check them, then scan.</div>`
    : `<div class="note warn">Loaded ${n} of 3 boxes from <b>${esc(bbox.name)}</b>; the rest fall outside this page. Draw the missing ones.</div>`;
  renderTools(); drawBoxes();
}
$("#mk-json").onchange = async e => {
  const f = e.target.files[0]; e.target.value = "";
  if (!f || !SC.info) return;
  try {
    const b = await readBoxJson(f), p = pg();
    if (SC.info.image && b.width && b.height && (b.width !== p.w || b.height !== p.h))
      throw new Error(`This JSON is for a ${fmt(b.width)} × ${fmt(b.height)} image but ${SC.info.name} is ${fmt(p.w)} × ${fmt(p.h)}. Check that ${b.name} belongs to this image.`);
    SC.bbox = b; applyBoxJson(b);
  }
  catch (err) { $("#mk-msg").innerHTML = `<div class="note bad">${esc(err.message)}</div>`; }
};
document.querySelectorAll("[data-go]").forEach(b => b.onclick = () => setView(b.dataset.go));

async function uploadFile(f, bbox = null) {
  const el = $("#scan-status");
  el.innerHTML = `<div class="note"><span class="spin"></span>Uploading and rendering <b>${esc(f.name)}</b>…</div>`;
  try {
    const r = await fetch("api/upload?name=" + encodeURIComponent(f.name), {method: "POST", body: f});
    const info = await r.json();
    if (info.error) throw new Error(info.error);
    // A TIFF/PNG/JPEG is used pixel for pixel, so its JSON must describe exactly this image (PDF pages are scaled instead).
    const p0 = info.pages[0];
    if (bbox && info.image && bbox.width && bbox.height && (bbox.width !== p0.w || bbox.height !== p0.h)) {
      fetch("api/delete/" + info.id, {method: "POST"});
      throw new Error(`This JSON is for a ${fmt(bbox.width)} × ${fmt(bbox.height)} image but the ${f.name.split(".").pop().toUpperCase()} is ${fmt(p0.w)} × ${fmt(p0.h)}. `
        + `Check that ${bbox.name} belongs to ${f.name}.`);
    }
    SC.info = info; SC.page = 0; SC.boxes = {}; SC.tool = TOOLS[0][0]; SC.bbox = bbox;
    startMarking();
    el.innerHTML = "";
  } catch (e) {
    el.innerHTML = `<div class="note bad">${esc(e.message)}</div>`;
  }
  $("#file").value = "";
}

function startMarking() {
  const info = SC.info;
  $("#scan-start").hidden = true; $("#mark").hidden = false;
  $("#mk-name").textContent = info.name;
  $("#mk-page-l").hidden = info.pages.length < 2;
  $("#mk-page").innerHTML = info.pages.map((p, i) => `<option value="${i}">${i + 1} of ${info.pages.length}</option>`).join("");
  $("#mk-page").value = SC.page;
  loadPage();
}
$("#mk-page").onchange = e => { SC.page = +e.target.value; SC.boxes = {}; SC.tool = TOOLS[0][0]; loadPage(); };
$("#mk-restart").onclick = () => { SC.info = null; showScan(); };

function pg() { return SC.info.pages[SC.page]; }
function loadPage() {
  const p = pg(), img = $("#mark-img");
  $("#mk-meta").textContent = `${fmt(p.w)} × ${fmt(p.h)} px${p.dpi ? ` · ${p.dpi} DPI` : ""}`;
  img.onload = () => { fitZoom(); drawBoxes(); };
  img.src = p.preview + "?t=" + SC.info.uploadedAt;
  $("#mk-msg").innerHTML = "";
  if (SC.bbox) applyBoxJson(SC.bbox);
  renderTools();
}
function fitZoom(tries = 0) {  // start at the width of the viewer
  const view = $("#mark-view");
  // The viewer may not be laid out yet when a big preview finishes loading: wait a frame and try again.
  if (view.clientWidth < 100 && tries < 120) return requestAnimationFrame(() => { fitZoom(tries + 1); drawBoxes(); });
  const w = Math.min(view.clientWidth - 32, $("#mark-img").naturalWidth || 1000);
  SC.baseW = w; $("#mk-zoom").value = 1; $("#mark-wrap").style.width = w + "px";
}
$("#mk-zoom").oninput = e => { const v = $("#mark-view"), wrap = $("#mark-wrap"), f = v.scrollTop / (wrap.offsetHeight || 1);
  wrap.style.width = SC.baseW * +e.target.value + "px"; v.scrollTop = f * wrap.offsetHeight; drawBoxes(); };

function renderTools() {
  $("#mk-tools").innerHTML = TOOLS.map(([k, name, col, help], i) => `<div class="tool ${SC.tool === k ? "on" : ""}" data-k="${k}"><i style="background:${col}"></i>
    <div><b>${i + 1}. ${name} ${SC.boxes[k] ? '<span class="ok">✓</span>' : ""}</b>${help}</div></div>`).join("");
  $("#mk-tools").querySelectorAll(".tool").forEach(t => t.onclick = () => { SC.tool = t.dataset.k; renderTools(); drawBoxes(); });
  $("#mk-scan").disabled = SC.busy || TOOLS.some(([k]) => !SC.boxes[k]);
}
const k2d = () => $("#mark-wrap").clientWidth / pg().w;   // page px -> display px
function drawBoxes() {
  const k = k2d();
  $("#mark-boxes").innerHTML = TOOLS.filter(([key]) => SC.boxes[key]).map(([key, name, col]) => {
    const [x0, y0, x1, y1] = SC.boxes[key];
    return `<div class="mbox ${SC.tool === key ? "act" : ""}" style="left:${x0 * k}px;top:${y0 * k}px;width:${(x1 - x0) * k}px;height:${(y1 - y0) * k}px;border-color:${col}"><span style="background:${col}">${name}</span></div>`;
  }).join("");
}

// drawing: new box, or mdrag an edge of an existing one; auto-scroll near the viewer's edges
const wrap = $("#mark-wrap"), view = $("#mark-view");
let mdrag = null;
function toPage(e) { const r = wrap.getBoundingClientRect(), k = k2d();
  return [Math.max(0, Math.min(pg().w, (e.clientX - r.left) / k)), Math.max(0, Math.min(pg().h, (e.clientY - r.top) / k))]; }
function edgeHit(e) {  // [toolKey, edges] if the pointer is on a box edge
  const [x, y] = toPage(e), tol = 7 / k2d();
  const order = [SC.tool, ...TOOLS.map(t => t[0]).filter(t => t !== SC.tool)];
  for (const key of order) { const b = SC.boxes[key]; if (!b) continue;
    const inX = x > b[0] - tol && x < b[2] + tol, inY = y > b[1] - tol && y < b[3] + tol;
    const ed = [inY && Math.abs(x - b[0]) < tol, inX && Math.abs(y - b[1]) < tol, inY && Math.abs(x - b[2]) < tol, inX && Math.abs(y - b[3]) < tol];
    if (ed.some(Boolean)) return [key, ed]; }
  return null;
}
wrap.addEventListener("pointermove", e => { if (mdrag) return; const h = edgeHit(e);
  wrap.style.cursor = !h ? "crosshair" : (h[1][0] || h[1][2]) && (h[1][1] || h[1][3]) ? "move" : h[1][0] || h[1][2] ? "ew-resize" : "ns-resize"; });
wrap.addEventListener("pointerdown", e => {
  if (!SC.info || e.button !== 0) return;
  e.preventDefault(); wrap.setPointerCapture(e.pointerId);
  const h = edgeHit(e), p = toPage(e);
  if (h) { SC.tool = h[0]; mdrag = {mode: "edge", key: h[0], ed: h[1], last: e}; }
  else { mdrag = {mode: "new", key: SC.tool, start: p, last: e}; SC.boxes[SC.tool] = [p[0], p[1], p[0], p[1]]; }
  renderTools(); drawBoxes(); autoScroll();
});
wrap.addEventListener("pointermove", e => { if (mdrag) { mdrag.last = e; applyDrag(); } });
wrap.addEventListener("pointerup", () => {
  if (!mdrag) return;
  const key = mdrag.key, b = SC.boxes[key];
  const n = [Math.min(b[0], b[2]), Math.min(b[1], b[3]), Math.max(b[0], b[2]), Math.max(b[1], b[3])];
  if (n[2] - n[0] < 6 || n[3] - n[1] < 6) delete SC.boxes[key];   // a click, not a box
  else {
    SC.boxes[key] = n.map(Math.round);
    // The two columns cover the same depths: snap top/bottom to the other column when close.
    const other = {depth_column: "interpreted_lithology_column", interpreted_lithology_column: "depth_column"}[key], o = SC.boxes[other];
    if (o) { const tol = pg().h * 0.03; if (Math.abs(n[1] - o[1]) < tol) SC.boxes[key][1] = o[1]; if (Math.abs(n[3] - o[3]) < tol) SC.boxes[key][3] = o[3]; }
    if (mdrag.mode === "new") { const next = TOOLS.find(([k]) => !SC.boxes[k]); if (next) SC.tool = next[0]; }
  }
  mdrag = null; renderTools(); drawBoxes();
});
function applyDrag() {
  const p = toPage(mdrag.last), b = SC.boxes[mdrag.key];
  if (mdrag.mode === "new") SC.boxes[mdrag.key] = [mdrag.start[0], mdrag.start[1], p[0], p[1]];
  else mdrag.ed.forEach((on, i) => { if (on) b[i] = p[i % 2]; });
  drawBoxes();
}
function autoScroll() {
  if (!mdrag) return;
  const r = view.getBoundingClientRect(), y = mdrag.last.clientY, edge = 50;
  const dy = y > r.bottom - edge ? (y - (r.bottom - edge)) / 2 : y < r.top + edge ? -((r.top + edge) - y) / 2 : 0;
  if (dy) { view.scrollTop += dy; applyDrag(); }
  requestAnimationFrame(autoScroll);
}

$("#mk-scan").onclick = async () => {
  const msg = $("#mk-msg");
  SC.busy = true; renderTools();
  msg.innerHTML = `<span class="spin"></span>Scanning… reading the depth numbers, the legend and the rock patterns. This usually takes 5 to 60 seconds.`;
  try {
    const r = await fetch("api/scan/" + SC.info.id, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({page: SC.page, boxes: SC.boxes,
      jsonSize: SC.bbox && SC.bbox.width && SC.bbox.height && SC.info.image ? [SC.bbox.width, SC.bbox.height] : null})});
    const m = await r.json();
    if (m.error) throw new Error(m.error);
    msg.innerHTML = `✓ Done: ${m.iv.length} intervals. Opening your report…`;
    location.hash = "mine=" + m.id; location.reload();
  } catch (e) {
    SC.busy = false; renderTools();
    msg.innerHTML = `<div class="note bad"><b>The scan didn't work.</b> ${esc(e.message)}<br>Check that the depth box covers the depth numbers and the rock-column box covers the same depths, then try again.</div>`;
  }
};

// =============================================================== SCANNED LOGS: a library just for uploaded logs
const MY = {cur: null, map: null, placing: false};
const kmBetween = (a, b) => { const R = 6371, r = Math.PI / 180, dLa = (b[0] - a[0]) * r, dLo = (b[1] - a[1]) * r;
  const h = Math.sin(dLa / 2) ** 2 + Math.cos(a[0] * r) * Math.cos(b[0] * r) * Math.sin(dLo / 2) ** 2; return 2 * R * Math.asin(Math.sqrt(h)); };

function renderMine() {
  const el = $("#mine"), ups = LOGS.filter(l => l.upload).sort((a, b) => b.uploadedAt - a.uploadedAt);
  if (MY.map) { MY.map.remove(); MY.map = null; }
  if (!ups.length) {
    el.innerHTML = `<div class="mine-top"><div><h2>Scanned logs</h2><p>No scans yet. Upload a mud log PDF with its bounding-box JSON to see its results here.</p></div></div>
      <button class="primary" data-go="scan">Scan a log</button>`;
    el.querySelector("[data-go]").onclick = () => setView("scan"); return;
  }
  const l = byId[MY.cur] || ups[0]; MY.cur = l.id;
  const d = l.dec, loc = WELL_LOC[l.well], tot = l.len || 1, foot = l.foot;
  const cats = Object.entries(foot).sort((a, b) => b[1] - a[1]);
  const tags = [d.area && d.offshore && d.area, d.block, d.lease, d.operator, d.type, d.scale && d.scale + "-inch", d.ref].filter(Boolean);
  const date = new Date(l.uploadedAt).toLocaleString(undefined, {dateStyle: "medium", timeStyle: "short"});
  let near = [];
  if (loc) near = Object.entries(WELL_LOC).filter(([k]) => k !== l.well && wells[k] && !wells[k][0].upload)
    .map(([k, v]) => [k, kmBetween(loc, v)]).sort((a, b) => a[1] - b[1]).slice(0, 4);
  el.innerHTML = `
    <div class="mine-top"><div><h2>Scanned logs</h2></div>
      <button class="primary" data-go="scan">+ Scan another log</button></div>
    ${ups.length > 1 ? `<div class="mine-pills">${ups.map(u => `<span class="chip ${u === l ? "on" : ""}" data-id="${u.id}">${esc(u.dec.title)}</span>`).join("")}</div>` : ""}
    <div class="mine-grid"><div>
      <div class="mcard">
        <h3>${esc(d.title)} <span class="status ${l.status[0]}">${l.status[1]}</span></h3>
        <div class="hint" style="margin-top:4px">${esc(l.file)}${l.pages > 1 ? ` · page ${l.page} of ${l.pages}` : ""} · scanned ${date}${l.dpi ? ` · ${l.dpi} DPI` : ""} · ${l.pipelineSeconds}s</div>
        <div class="fname" title="Hover the coloured parts to decode the file name">${fnameHTML(d)}</div>
        ${tags.length ? `<div class="tags" style="margin-top:8px">${tags.map(t => `<span class="tag">${esc(t)}</span>`).join(" ")}</div>` : ""}
        <div class="acts">
          <button class="primary" id="my-open">Open the log report</button>
          <button id="my-map" ${loc ? "" : "disabled title='Place it on the map first'"}>See it in 3D among the other wells</button>
          <a href="${l.csv}" download>Download CSV</a>
          ${l.overlay ? `<a href="${l.overlay}" target="_blank">Pipeline overlay image</a>` : ""}
          <button id="my-del" style="margin-left:auto;color:#b42318" ${SERVER ? "" : "disabled title='Needs the scanner running'"}>Delete scan</button>
        </div>
      </div>
      <div class="mcard"><h4>What the scanner found</h4>
        <div class="kpi-row"><div><b>${fmt(l.top)}–${fmt(l.base)}</b>ft logged</div><div><b>${l.iv.length}</b>intervals</div>
          <div><b>${cats.filter(([c]) => !/Unmatched|Annotation|Other/.test(c)).length}</b>rock types</div><div><b>${(l.fit || {}).labels_used || 0}</b>depth numbers read</div></div>
        <div class="stack" style="height:18px">${cats.map(([c, f]) => `<i title="${c}: ${fmt(f / tot * 100)}%" style="width:${f / tot * 100}%;background:${COL[c]}"></i>`).join("")}</div>
        <div class="mix">${cats.map(([c, f]) => `<span><i class="sw" style="background:${COL[c]};width:11px;height:11px;border:0"></i>${c}<em>${fmt(f / tot * 100)}% · ${ft(f)}</em></span>`).join("")}</div>
      </div>
      <div class="mcard"><h4>Intervals · click one to open it on the scan</h4><div class="ivt-wrap"><table class="ivt">
        <tr><th>Top (ft)</th><th>Base (ft)</th><th>Thickness</th><th>Rock</th><th>Confidence</th></tr>
        ${l.iv.map((v, i) => `<tr data-i="${i}"><td>${fmt(v.t, 1)}</td><td>${fmt(v.b, 1)}</td><td>${fmt(v.b - v.t, 1)} ft</td>
          <td><i class="sw" style="background:${COL[v.cat]};width:10px;height:10px;border:0;margin-right:5px"></i>${esc(v.lith)}${v.amb ? ` <span class="hint">(or ${esc(v.alt[0] || "?")})</span>` : ""}</td>
          <td><div class="cbar" style="display:inline-block;width:60px;vertical-align:middle"><i style="width:${v.c * 100}%;background:${confColour(v.c)}"></i></div> ${fmt(v.c * 100)}%</td></tr>`).join("")}
      </table></div></div>
    </div><div>
      <div class="mcard"><h4>Where it sits</h4><div id="mine-map"></div>
        <div style="margin-top:10px;font-size:13px;line-height:1.5">${loc
          ? `<b>${esc(loc[3])}</b><br><span class="hint">${loc[0].toFixed(3)}°N, ${Math.abs(loc[1]).toFixed(3)}°W · ${PREC[loc[2]]}</span>`
          : `<b>We couldn't tell where this well is from its file name.</b><br><span class="hint">Enter its lease block below, or click “Drop a pin” and click the map.</span>`}</div>
        ${near.length ? `<h4 style="margin-top:14px">Nearest wells in the library</h4>${near.map(([k, km]) => `<div class="near" data-w="${k}"><span>${esc(wellTitle(k))}</span><span class="hint">${fmt(km)} km · ${wells[k].length} log${wells[k].length > 1 ? "s" : ""}</span></div>`).join("")}` : ""}
        <h4 style="margin-top:14px">Set the location yourself</h4>
        <div class="place"><select id="pl-area">${Object.keys(BLOCK_REF).map(c => `<option value="${c}" ${d.block && d.block.startsWith(c) ? "selected" : ""}>${BLOCK_AREA[c]}</option>`).join("")}</select>
          block <input id="pl-block" type="number" min="1" max="1100" value="${d.block ? d.block.split(" ")[1] : ""}"><button id="pl-go">Place</button></div>
        <div class="place"><button id="pl-pin">Drop a pin on the map</button>${l.loc ? `<button id="pl-auto" class="link">Go back to the automatic position</button>` : ""}</div>
        ${SERVER ? "" : `<div class="hint" style="margin-top:6px">Saving a location needs the scanner running (python3 server.py).</div>`}
        <div id="pl-msg" class="hint"></div>
      </div>
      <div class="mcard"><h4>Legend read from the scan</h4>
        <div class="tags">${l.legend.length ? l.legend.map(n => `<span class="tag">${esc(n)}</span>`).join(" ") : '<span class="hint">No legend symbols found.</span>'}</div>
        ${l.hasLegendImg ? `<img class="legend-img" src="data/legend/${l.id}.jpg" alt="legend crop">` : ""}
      </div>
    </div></div>`;

  el.querySelectorAll("[data-go]").forEach(b => b.onclick = () => setView(b.dataset.go));
  el.querySelectorAll(".mine-pills .chip").forEach(c => c.onclick = () => { MY.cur = c.dataset.id; history.replaceState(null, "", "#mine=" + MY.cur); renderMine(); });
  $("#my-open").onclick = () => openLog(l);
  $("#my-map").onclick = () => { E.sel = l.well; setView("earth"); setEarthMode("3d"); selectWell(l.well); };
  el.querySelectorAll(".ivt tr[data-i]").forEach(r => r.onclick = () => openLog(l, +r.dataset.i));
  el.querySelectorAll(".near").forEach(r => r.onclick = () => { E.sel = r.dataset.w; setView("earth"); selectWell(r.dataset.w); });
  $("#my-del").onclick = async () => {
    if (!confirm(`Delete the scan of ${l.file}?`)) return;
    await fetch("api/delete/" + l.id, {method: "POST"}); location.hash = "mine"; location.reload();
  };
  const save = async body => {
    if (!SERVER) return;
    $("#pl-msg").textContent = "Saving…";
    const r = await (await fetch("api/locate/" + l.id, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)})).json();
    if (r.error) { $("#pl-msg").textContent = r.error; return; }
    location.hash = "mine=" + l.id; location.reload();   // reload so every view (3D map included) picks up the new position
  };
  $("#pl-go").onclick = () => { const c = $("#pl-area").value, n = +$("#pl-block").value, p = blockLoc(c, n);
    if (!p) { $("#pl-msg").textContent = "Enter a block number."; return; }
    save({lat: p[0], lon: p[1], precision: "grid", desc: `${BLOCK_AREA[c]} block ${n}, entered by you`}); };
  $("#pl-pin").onclick = () => { MY.placing = !MY.placing; $("#pl-pin").classList.toggle("on", MY.placing);
    $("#pl-msg").textContent = MY.placing ? "Click the map where the well is." : ""; $("#mine-map").style.cursor = MY.placing ? "crosshair" : ""; };
  if ($("#pl-auto")) $("#pl-auto").onclick = () => save({clear: true});

  // mini map: the library's wells in grey, this scan as a gold pin
  const mm = MY.map = L.map("mine-map", {zoomControl: true, attributionControl: false, scrollWheelZoom: false});
  if (loc) mm.setView([loc[0], loc[1]], loc[2] === "area" ? 5 : 6); else mm.fitBounds([[25.5, -97], [30.5, -87]]);   // view first: Leaflet needs it before vector layers
  const world = topojson.feature(WORLD, WORLD.objects.countries);
  const land = L.geoJSON(world, {style: {color: "#9aa6ad", weight: .6, fillColor: "#eef0e6", fillOpacity: 1}, interactive: false}).addTo(mm);
  const ocean = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}", {maxZoom: 13}).addTo(mm);
  ocean.once("tileload", () => mm.removeLayer(land));
  PROTRACTION.forEach(([n, la0, la1, lo0, lo1]) => L.rectangle([[la0, lo0], [la1, lo1]], {color: "#2b5d8c", weight: 1, dashArray: "4 4", fill: false, interactive: false}).addTo(mm)
    .bindTooltip(n, {permanent: false}));
  Object.entries(WELL_LOC).forEach(([k, v]) => { if (!wells[k] || k === l.well) return;
    L.circleMarker([v[0], v[1]], {radius: wells[k][0].upload ? 6 : 5, color: "#fff", weight: 1.5, fillColor: wells[k][0].upload ? "#e0a800" : "#2b5d8c", fillOpacity: .9})
      .bindTooltip(`<b>${esc(wellTitle(k))}</b><br>${wells[k].length} log${wells[k].length > 1 ? "s" : ""}`).on("click", () => { E.sel = k; setView("earth"); selectWell(k); }).addTo(mm); });
  if (loc) {
    const r = 18;
    L.marker([loc[0], loc[1]], {icon: L.divIcon({className: "pin up", html: donut(foot, r), iconSize: [2 * r, 2 * r], iconAnchor: [r, r]}), zIndexOffset: 1000})
      .bindTooltip(`<b>★ ${esc(d.title)}</b><br>Scanned log`, {direction: "top", offset: [0, -r]}).addTo(mm);
    if (loc[2] === "area") { const a = PROTRACTION.find(x => loc[3].includes(x[0])); if (a) L.rectangle([[a[1], a[3]], [a[2], a[4]]], {color: "#e0a800", weight: 2, fillOpacity: .08}).addTo(mm); }
  }
  mm.on("click", e => { if (MY.placing) save({lat: +e.latlng.lat.toFixed(4), lon: +e.latlng.lng.toFixed(4), precision: "user", desc: "Placed by you on the map"}); });
}

// =============================================================== boot routing
(() => {
  const h = new URLSearchParams(location.hash.slice(1));
  $("#tab-mine").style.display = UP.length ? "" : "none";
  if (h.has("mine") && UP.length) { MY.cur = h.get("mine"); setView("mine"); }
  else setView("scan");
})();
