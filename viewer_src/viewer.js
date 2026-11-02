// Mequa 3D viewer: a Streamlit custom component (no build step needed at runtime; this file is bundled
// into ../mequa/viewer/web/viewer.bundle.js by build.sh). Orbit, 3D box-select, measure, five
// representations, rest-to-loaded animation, clip plane, PNG export. Talks to Streamlit with the
// plain postMessage protocol.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ConvexGeometry } from "three/examples/jsm/geometries/ConvexGeometry.js";

let MINT = 0x0affd2, INK = 0x0a0a0a;
const THEMES = { light: { bg: 0xffffff, ink: 0x0a0a0a, mint: 0x0affd2, grid: [0xdddddd, 0xeeeeee], cat0: 0x0a0a0a, low: 0xcfd6de }, green: { bg: 0x060a08, ink: 0xe6f2ea, mint: 0x22e58b, grid: [0x1d2a22, 0x111a15], cat0: 0xe6f2ea, low: 0x3a4a41 } };
let gridObj = null;
function setTheme(name) {
  const t = THEMES[name] || THEMES.light; if (S._theme === name) return; S._theme = name;
  document.body.classList.toggle("green", name === "green"); INK = t.ink; MINT = t.mint; S.bg = t.bg; CAT[0] = t.cat0; STOPS.strain[0] = t.low; STOPS.strain[1] = t.mint;
  if (typeof scene3 !== "undefined" && scene3) { scene3.background = new THREE.Color(t.bg); scene3.fog = new THREE.Fog(t.bg, 14, 38); if (gridObj) { scene3.remove(gridObj); gridObj = new THREE.GridHelper(14, 14, t.grid[0], t.grid[1]); gridObj.rotation.x = Math.PI / 2; gridObj.position.z = -5.4; scene3.add(gridObj); }
    if (S.scene) { applyColors(); } }
}
const $ = (id) => document.getElementById(id);
const send = (type, data = {}) => window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*");
const setValue = (value) => send("streamlit:setComponentValue", { value, dataType: "json" });

const S = {
  scene: null, sceneId: null, selRev: null, selected: new Set(), nonce: 0, rep: "spheres", color: "uniform", tool: "orbit",
  t: 1, playing: false, dir: 1, spin: false, size: 1, vectors: false, clipAxis: "off", clipPos: 0.5, ortho: false,
  measure: [], hover: -1, cmap: "strain", ready: false, bg: 0xffffff, n: 0,
};
let renderer, scene3, persp, ortho, camera, controls, group, raycaster, gizmoScene, gizmoCam, clip;
let rest, loaded, cur, off = new THREE.Vector3(), scl = 1, edges = null, nnDist = 1, mesh = null, halo = null, linesObj = null, sticks = null, hullObj = null, ptsObj = null, vecObj = null, measObj = null;
const clipPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 0);

// ----------------------------------------------------------------------------------- colour
function lerp3(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
const hex = (h) => [(h >> 16 & 255) / 255, (h >> 8 & 255) / 255, (h & 255) / 255];
const STOPS = { strain: [0xcfd6de, 0x0affd2, 0xfacc15, 0xef4444], viridis: [0x440154, 0x3b528b, 0x21918c, 0x5ec962, 0xfde725], ink: [0xeeeeee, 0x8a8a8a, 0x0a0a0a] };
function ramp(stops, t) { t = Math.min(1, Math.max(0, t)); const k = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(k)); return lerp3(hex(stops[i]), hex(stops[i + 1]), k - i); }
const CAT = [0x0a0a0a, 0x0affd2, 0x3b82f6, 0xef4444, 0xfacc15, 0x9b7fd1, 0x4caf7a, 0x8a8a8a];

function fieldInfo(name) {
  const f = S.scene.fields && S.scene.fields[name]; if (!f) return null;
  const kind = (S.scene.field_kinds || {})[name] || "continuous";
  let lo = Infinity, hi = -Infinity; for (let i = 0; i < f.length; i++) { if (f[i] < lo) lo = f[i]; if (f[i] > hi) hi = f[i]; }
  return { f, kind, lo, hi };
}
function colorOf(i) {
  const sc = S.scene;
  if (S.color === "element" && sc.colors) return hex(parseInt(sc.colors[i].replace("#", ""), 16));
  if (S.color === "uniform" || !sc.fields || !sc.fields[S.color]) return hex(sc.colors ? parseInt(sc.colors[i].replace("#", ""), 16) : 0x7d8798);
  const fi = S._fi; if (!fi) return hex(0x7d8798);
  if (fi.kind === "categorical") return hex(CAT[((fi.f[i] | 0) % CAT.length + CAT.length) % CAT.length]);
  const t = fi.hi > fi.lo ? (fi.f[i] - fi.lo) / (fi.hi - fi.lo) : 0.5;
  return ramp(S.color === "stiffness" ? STOPS.viridis : STOPS.strain, t);
}
function drawLegend() {
  const el = $("legend"); const sc = S.scene; el.innerHTML = "";
  if (S.color === "uniform" || S.color === "element" || !S._fi) { el.style.display = "none"; return; }
  el.style.display = "block"; const fi = S._fi;
  if (fi.kind === "categorical") {
    const ids = [...new Set(Array.from(fi.f))].sort((a, b) => a - b).slice(0, 8);
    el.innerHTML = `<b>${S.color}</b>` + ids.map((v) => `<span class="chip"><i style="background:#${CAT[((v | 0) % CAT.length + CAT.length) % CAT.length].toString(16).padStart(6, "0")}"></i>${v | 0}</span>`).join("");
  } else {
    const st = (S.color === "stiffness" ? STOPS.viridis : STOPS.strain).map((h) => "#" + h.toString(16).padStart(6, "0")).join(",");
    el.innerHTML = `<b>${S.color}</b><div class="bar" style="background:linear-gradient(90deg,${st})"></div><span>${fi.lo.toPrecision(3)}</span><span style="float:right">${fi.hi.toPrecision(3)}</span>`;
  }
}

// ----------------------------------------------------------------------------------- scene build
function medianNN(P, n) {
  const m = Math.min(n, 300), step = Math.max(1, Math.floor(n / m)), ds = [];
  for (let a = 0; a < n; a += step) {
    let best = Infinity;
    for (let b = 0; b < n; b++) { if (b === a) continue; const dx = P[3 * a] - P[3 * b], dy = P[3 * a + 1] - P[3 * b + 1], dz = P[3 * a + 2] - P[3 * b + 2]; const d = dx * dx + dy * dy + dz * dz; if (d < best) best = d; }
    ds.push(Math.sqrt(best));
  }
  ds.sort((x, y) => x - y); return ds[ds.length >> 1] || 1;
}
function clear() {
  for (const o of [mesh, halo, linesObj, sticks, hullObj, ptsObj, vecObj, measObj]) if (o) { group.remove(o); o.geometry && o.geometry.dispose(); }
  mesh = halo = linesObj = sticks = hullObj = ptsObj = vecObj = measObj = null;
}
function buildScene(sc, refit) {
  S.scene = sc; const n = (S.n = sc.n);
  rest = Float32Array.from(sc.pos); loaded = sc.loaded ? Float32Array.from(sc.loaded) : null; cur = new Float32Array(rest);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) { const v = rest[3 * i + k]; if (v < lo[k]) lo[k] = v; if (v > hi[k]) hi[k] = v; }
  const ext = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-9);
  off.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2); scl = 10 / ext;
  const norm = (A) => { for (let i = 0; i < n; i++) { A[3 * i] = (A[3 * i] - off.x) * scl; A[3 * i + 1] = (A[3 * i + 1] - off.y) * scl; A[3 * i + 2] = (A[3 * i + 2] - off.z) * scl; } };
  norm(rest); if (loaded) norm(loaded);
  edges = sc.edges && sc.edges.length ? Uint32Array.from(sc.edges) : null;
  nnDist = n > 1 ? medianNN(rest, n) : 1;
  S._fi = fieldInfo(S.color);
  if (!S.ready || refit) { camera.position.set(9, 7, 11); controls.target.set(0, 0, 0); controls.update(); }
  S.ready = true; rebuildGeometry(); updateInfo(); drawLegend(); fillColorSelect();
  $("loadrow").style.display = loaded ? "flex" : "none";
}
function fillColorSelect() {
  const sel = $("color"), sc = S.scene; const opts = ["uniform"].concat(sc.colors ? ["element"] : []).concat(Object.keys(sc.fields || {}));
  sel.innerHTML = opts.map((o) => `<option ${o === S.color ? "selected" : ""}>${o}</option>`).join("");
}
function radiusOf(i) { const r = S.scene.radii ? S.scene.radii[i] * scl : 1; return r; }
function baseRadius() { return (S.rep === "ballstick" ? 0.16 : 0.33) * nnDist * S.size; }
function rebuildGeometry() {
  clear(); const n = S.n; const detail = n > 8000 ? 0 : n > 2500 ? 1 : 2;
  const mk = (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.05, clippingPlanes: clip });
  if (S.rep === "spheres" || S.rep === "ballstick") {
    mesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, detail), mk(0xffffff), n); mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); group.add(mesh);
    halo = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshBasicMaterial({ color: MINT, transparent: true, opacity: 0.55, depthWrite: false, clippingPlanes: clip }), Math.max(1, n)); group.add(halo);
  }
  if (S.rep === "ballstick" && edges) {
    const m = edges.length / 2;
    if (m <= 30000) { sticks = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 8, 1), new THREE.MeshStandardMaterial({ color: 0xb8bfc9, roughness: 0.6, clippingPlanes: clip }), m); sticks.instanceMatrix.setUsage(THREE.DynamicDrawUsage); group.add(sticks); }
  }
  if (S.rep === "links" || (S.rep === "ballstick" && !sticks) || S.rep === "hull") {
    if (edges && S.rep !== "hull") {
      const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(edges.length * 3), 3)); g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(edges.length * 3), 3));
      linesObj = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85, clippingPlanes: clip })); group.add(linesObj);
    }
  }
  if (S.rep === "points") {
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3)); g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    ptsObj = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.16 * S.size, vertexColors: true, sizeAttenuation: true, clippingPlanes: clip })); group.add(ptsObj);
    halo = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshBasicMaterial({ color: MINT, transparent: true, opacity: 0.6, depthWrite: false }), Math.max(1, n)); group.add(halo);
  }
  if (S.rep === "hull" && n >= 4 && n <= 30000) {
    const pts = []; for (let i = 0; i < n; i++) pts.push(new THREE.Vector3(rest[3 * i], rest[3 * i + 1], rest[3 * i + 2]));
    try { hullObj = new THREE.Mesh(new ConvexGeometry(pts), new THREE.MeshStandardMaterial({ color: 0x0affd2, transparent: true, opacity: 0.22, side: THREE.DoubleSide, roughness: 0.3, depthWrite: false, clippingPlanes: clip })); group.add(hullObj); } catch (e) { /* degenerate (coplanar) set */ }
    ptsObj = new THREE.Points(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3)).setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3)), new THREE.PointsMaterial({ size: 0.12 * S.size, vertexColors: true, clippingPlanes: clip })); group.add(ptsObj);
  }
  vecObj = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 6), 3)), new THREE.LineBasicMaterial({ color: INK, transparent: true, opacity: 0.55 })); vecObj.visible = S.vectors && !!loaded; group.add(vecObj);
  measObj = new THREE.Group(); group.add(measObj); measObj.geometry = null;
  applyPositions(); applyColors(); applySelection();
}
const tmp = new THREE.Object3D(), up = new THREE.Vector3(0, 1, 0), qa = new THREE.Quaternion(), va = new THREE.Vector3(), vb = new THREE.Vector3();
function applyPositions() {
  const n = S.n, t = loaded ? S.t : 0;
  for (let i = 0; i < 3 * n; i++) cur[i] = loaded ? rest[i] + (loaded[i] - rest[i]) * t : rest[i];
  const br = baseRadius();
  if (mesh) {
    for (let i = 0; i < n; i++) { const r = S.scene.radii ? radiusOf(i) * (S.rep === "ballstick" ? 0.32 : 0.9) * S.size : br; tmp.position.set(cur[3 * i], cur[3 * i + 1], cur[3 * i + 2]); tmp.scale.setScalar(r); tmp.updateMatrix(); mesh.setMatrixAt(i, tmp.matrix); }
    mesh.instanceMatrix.needsUpdate = true; mesh.computeBoundingSphere();
  }
  if (sticks) {
    const m = edges.length / 2;
    for (let e = 0; e < m; e++) { const a = edges[2 * e], b = edges[2 * e + 1]; va.set(cur[3 * a], cur[3 * a + 1], cur[3 * a + 2]); vb.set(cur[3 * b], cur[3 * b + 1], cur[3 * b + 2]);
      const d = vb.clone().sub(va), L = d.length() || 1e-6; qa.setFromUnitVectors(up, d.normalize()); tmp.position.copy(va).add(vb).multiplyScalar(0.5); tmp.quaternion.copy(qa); tmp.scale.set(br * 0.45, L, br * 0.45); tmp.updateMatrix(); sticks.setMatrixAt(e, tmp.matrix); }
    sticks.instanceMatrix.needsUpdate = true; tmp.quaternion.identity();
  }
  if (linesObj) { const p = linesObj.geometry.attributes.position.array; for (let k = 0; k < edges.length; k++) { const v = edges[k]; p[3 * k] = cur[3 * v]; p[3 * k + 1] = cur[3 * v + 1]; p[3 * k + 2] = cur[3 * v + 2]; } linesObj.geometry.attributes.position.needsUpdate = true; linesObj.geometry.computeBoundingSphere(); }
  if (ptsObj) { ptsObj.geometry.attributes.position.array.set(cur); ptsObj.geometry.attributes.position.needsUpdate = true; ptsObj.geometry.computeBoundingSphere(); }
  if (vecObj && loaded) { const p = vecObj.geometry.attributes.position.array; for (let i = 0; i < n; i++) { p[6 * i] = rest[3 * i]; p[6 * i + 1] = rest[3 * i + 1]; p[6 * i + 2] = rest[3 * i + 2]; p[6 * i + 3] = cur[3 * i]; p[6 * i + 4] = cur[3 * i + 1]; p[6 * i + 5] = cur[3 * i + 2]; } vecObj.geometry.attributes.position.needsUpdate = true; }
  applySelection(true); drawMeasure();
}
function applyColors() {
  S._fi = fieldInfo(S.color); const n = S.n, c = new THREE.Color();
  if (mesh) { for (let i = 0; i < n; i++) { const [r, g, b] = colorOf(i); c.setRGB(r, g, b); mesh.setColorAt(i, c); } mesh.instanceColor.needsUpdate = true; }
  if (ptsObj) { const a = ptsObj.geometry.attributes.color.array; for (let i = 0; i < n; i++) { const [r, g, b] = colorOf(i); a[3 * i] = r; a[3 * i + 1] = g; a[3 * i + 2] = b; } ptsObj.geometry.attributes.color.needsUpdate = true; }
  if (linesObj) {
    const a = linesObj.geometry.attributes.color.array, es = S.scene.edge_strain; let lo = Infinity, hi = -Infinity;
    if (es) for (const v of es) { if (v < lo) lo = v; if (v > hi) hi = v; }
    for (let e = 0; e < edges.length / 2; e++) { let col; if (es && S.color !== "uniform") col = ramp(STOPS.strain, hi > lo ? (es[e] - lo) / (hi - lo) : 0.5); else col = hex(0x9aa3af); for (let s = 0; s < 2; s++) { const k = 6 * e + 3 * s; a[k] = col[0]; a[k + 1] = col[1]; a[k + 2] = col[2]; } }
    linesObj.geometry.attributes.color.needsUpdate = true;
  }
  drawLegend();
}
function applySelection(posOnly) {
  const ids = [...S.selected]; const n = ids.length;
  if (halo && halo.isInstancedMesh) {
    halo.count = n; const br = baseRadius() * 1.55;
    for (let k = 0; k < n; k++) { const i = ids[k]; tmp.position.set(cur[3 * i], cur[3 * i + 1], cur[3 * i + 2]); tmp.scale.setScalar(S.scene.radii && S.rep !== "points" ? radiusOf(i) * 0.5 : S.rep === "points" ? 0.14 * S.size : br); tmp.updateMatrix(); halo.setMatrixAt(k, tmp.matrix); }
    halo.instanceMatrix.needsUpdate = true; halo.visible = n > 0;
  }
  $("selcount").textContent = n ? `${n} selected` : "none selected";
}

// ----------------------------------------------------------------------------------- measure and info
function worldOf(i) { return new THREE.Vector3(cur[3 * i], cur[3 * i + 1], cur[3 * i + 2]); }
function origOf(i) { return [cur[3 * i] / scl + off.x, cur[3 * i + 1] / scl + off.y, cur[3 * i + 2] / scl + off.z]; }
function label(text) {
  const cv = document.createElement("canvas"); cv.width = 256; cv.height = 64; const g = cv.getContext("2d");
  g.fillStyle = "rgba(255,255,255,.92)"; g.fillRect(0, 0, 256, 64); g.strokeStyle = "#0a0a0a"; g.strokeRect(0.5, 0.5, 255, 63); g.fillStyle = "#0a0a0a"; g.font = "600 28px ui-monospace,Consolas,monospace"; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(text, 128, 33);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), depthTest: false })); sp.scale.set(2.4, 0.6, 1); return sp;
}
function drawMeasure() {
  if (!measObj) return; while (measObj.children.length) { const c = measObj.children.pop(); c.geometry && c.geometry.dispose(); }
  const m = S.measure; if (m.length === 0) return;
  for (const i of m) { const s = new THREE.Mesh(new THREE.SphereGeometry(baseRadius() * 1.3, 12, 12), new THREE.MeshBasicMaterial({ color: INK })); s.position.copy(worldOf(i)); measObj.add(s); }
  if (m.length === 2) {
    const a = worldOf(m[0]), b = worldOf(m[1]); const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    measObj.add(new THREE.Line(g, new THREE.LineDashedMaterial({ color: INK, dashSize: 0.2, gapSize: 0.12 }))); measObj.children[measObj.children.length - 1].computeLineDistances();
    const A = origOf(m[0]), B = origOf(m[1]); const d = Math.hypot(A[0] - B[0], A[1] - B[1], A[2] - B[2]);
    const sp = label(d.toPrecision(4)); sp.position.copy(a).add(b).multiplyScalar(0.5); measObj.add(sp); S.lastDist = d;
  }
}
function updateInfo(i) {
  const el = $("info"); const sc = S.scene; if (!sc) return;
  let h = `<b>${sc.title || "model"}</b> · ${S.n} cells`;
  if (i !== undefined && i >= 0) {
    const p = origOf(i); h += `<br>#${i}${sc.labels ? " · " + sc.labels[i] : ""} · x ${p[0].toFixed(3)} y ${p[1].toFixed(3)} z ${p[2].toFixed(3)}`;
    for (const k of Object.keys(sc.fields || {})) h += ` · ${k} ${(+sc.fields[k][i]).toPrecision(4)}`;
  }
  if (S.measure.length === 2 && S.lastDist !== undefined) h += `<br>distance #${S.measure[0]}–#${S.measure[1]} = <b>${S.lastDist.toPrecision(5)}</b>`;
  else if (S.tool === "measure") h += `<br>measure: click ${S.measure.length === 0 ? "two cells" : "one more cell"}`;
  el.innerHTML = h;
}

// ----------------------------------------------------------------------------------- picking and box select
const ndc = new THREE.Vector2();
function pickAt(ev) {
  const r = renderer.domElement.getBoundingClientRect(); ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  raycaster.setFromCamera(ndc, camera); let target = mesh || null;
  if (target) { const hit = raycaster.intersectObject(target, false); if (hit.length) return hit[0].instanceId; return -1; }
  if (ptsObj) { raycaster.params.Points.threshold = 0.18 * S.size; const hit = raycaster.intersectObject(ptsObj, false); if (hit.length) return hit[0].index; }
  return -1;
}
function emit(mode) { S.nonce++; setValue({ ids: [...S.selected].sort((a, b) => a - b), nonce: S.nonce, source: "viewer", mode }); }
function boxSelect(x0, y0, x1, y1, mode) {
  const r = renderer.domElement.getBoundingClientRect(); const lx = Math.min(x0, x1), hx = Math.max(x0, x1), ly = Math.min(y0, y1), hy = Math.max(y0, y1);
  const v = new THREE.Vector3(); const hit = [];
  camera.updateMatrixWorld();
  for (let i = 0; i < S.n; i++) {
    v.set(cur[3 * i], cur[3 * i + 1], cur[3 * i + 2]); if (clip.length && clipPlane.distanceToPoint(v) < 0) continue;
    v.project(camera); if (v.z > 1) continue; const sx = (v.x * 0.5 + 0.5) * r.width, sy = (-v.y * 0.5 + 0.5) * r.height;
    if (sx >= lx && sx <= hx && sy >= ly && sy <= hy) hit.push(i);
  }
  if (mode === "replace") S.selected = new Set(hit); else if (mode === "add") hit.forEach((i) => S.selected.add(i)); else hit.forEach((i) => S.selected.delete(i));
  applySelection(); emit(mode);
}

// ----------------------------------------------------------------------------------- UI wiring
function setRep(r) { S.rep = r; document.querySelectorAll("[data-rep]").forEach((b) => b.classList.toggle("on", b.dataset.rep === r)); if (S.scene) rebuildGeometry(); }
function setTool(t) { S.tool = t; document.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === t)); controls.enabled = t !== "box"; renderer.domElement.style.cursor = t === "box" ? "crosshair" : t === "measure" ? "cell" : "grab"; if (t !== "measure") { S.measure = []; drawMeasure(); } updateInfo(); }
function setView(name) {
  const d = camera.position.distanceTo(controls.target) || 16; const dirs = { top: [0, 0, 1], front: [0, -1, 0], side: [1, 0, 0], iso: [0.55, -0.6, 0.55] };
  const v = new THREE.Vector3(...dirs[name]).normalize().multiplyScalar(d); camera.position.copy(controls.target).add(v); camera.up.set(0, name === "top" ? 1 : 0, name === "top" ? 0 : 1); if (name === "top") camera.up.set(0, 1, 0); camera.lookAt(controls.target); controls.update();
}
function fit(onlySel) {
  let ids = onlySel && S.selected.size ? [...S.selected] : null; const b = new THREE.Box3();
  for (let k = 0; k < (ids ? ids.length : S.n); k++) { const i = ids ? ids[k] : k; b.expandByPoint(worldOf(i)); }
  const c = b.getCenter(new THREE.Vector3()), s = b.getSize(new THREE.Vector3()).length() || 1; controls.target.copy(c);
  const dir = camera.position.clone().sub(controls.target).normalize(); camera.position.copy(c).add(dir.multiplyScalar(s * 1.25)); controls.update();
}
function setClip() {
  if (S.clipAxis === "off") { clip.length = 0; }
  else { const k = { x: 0, y: 1, z: 2 }[S.clipAxis]; const n = new THREE.Vector3(); n.setComponent(k, -1); clipPlane.normal.copy(n); const r = 5.2; clipPlane.constant = (S.clipPos * 2 - 1) * r; clip[0] = clipPlane; }
  renderer.localClippingEnabled = true;
  group.traverse((o) => { if (o.material) o.material.needsUpdate = true; });
}
function wire() {
  document.querySelectorAll("[data-rep]").forEach((b) => (b.onclick = () => setRep(b.dataset.rep)));
  document.querySelectorAll("[data-tool]").forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
  document.querySelectorAll("[data-view]").forEach((b) => (b.onclick = () => setView(b.dataset.view)));
  $("color").onchange = (e) => { S.color = e.target.value; applyColors(); };
  $("size").oninput = (e) => { S.size = +e.target.value; rebuildGeometry(); };
  $("t").oninput = (e) => { S.t = +e.target.value / 100; applyPositions(); };
  $("play").onclick = () => { S.playing = !S.playing; $("play").classList.toggle("on", S.playing); };
  $("vec").onchange = (e) => { S.vectors = e.target.checked; if (vecObj) vecObj.visible = S.vectors && !!loaded; };
  $("persp").onclick = () => { S.ortho = !S.ortho; const t = controls.target.clone(), p = camera.position.clone(); camera = S.ortho ? ortho : persp; camera.position.copy(p); controls.object = camera; controls.target.copy(t); camera.up.set(0, 0, 1); resize(); $("persp").textContent = S.ortho ? "Ortho" : "Persp"; };
  $("spin").onclick = () => { S.spin = !S.spin; controls.autoRotate = S.spin; $("spin").classList.toggle("on", S.spin); };
  $("fit").onclick = () => fit(false); $("fitsel").onclick = () => fit(true);
  $("clipaxis").onchange = (e) => { S.clipAxis = e.target.value; setClip(); }; $("clippos").oninput = (e) => { S.clipPos = +e.target.value / 100; setClip(); };
  $("shot").onclick = () => { render(); const a = document.createElement("a"); a.download = "mequa_view.png"; a.href = renderer.domElement.toDataURL("image/png"); a.click(); };
  $("clr").onclick = () => { S.selected = new Set(); S.measure = []; applySelection(); drawMeasure(); updateInfo(); emit("replace"); };
  $("inv").onclick = () => { const s = new Set(); for (let i = 0; i < S.n; i++) if (!S.selected.has(i)) s.add(i); S.selected = s; applySelection(); emit("replace"); };
  window.addEventListener("keydown", (e) => { if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return; const k = e.key.toLowerCase();
    if (k === "r") { camera.position.set(9, 7, 11); controls.target.set(0, 0, 0); controls.update(); } else if (k === " ") { e.preventDefault(); $("play").click(); } else if (k === "escape") $("clr").click(); else if (k === "b") setTool("box"); else if (k === "o") setTool("orbit"); else if (k === "m") setTool("measure"); else if (k === "f") fit(true); else if ("1234".includes(k) && k) setView(["top", "front", "side", "iso"][+k - 1]); });
  const el = renderer.domElement; let down = null; const rect = $("rect");
  el.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; if (S.tool === "box") { el.setPointerCapture(e.pointerId); const r = el.getBoundingClientRect(); down.rx = e.clientX - r.left; down.ry = e.clientY - r.top; rect.style.display = "block"; } });
  el.addEventListener("pointermove", (e) => {
    const r = el.getBoundingClientRect();
    if (down && S.tool === "box") { const x = e.clientX - r.left, y = e.clientY - r.top; rect.style.left = Math.min(x, down.rx) + "px"; rect.style.top = Math.min(y, down.ry) + "px"; rect.style.width = Math.abs(x - down.rx) + "px"; rect.style.height = Math.abs(y - down.ry) + "px"; return; }
    if (!down && S.scene && mesh) { const i = pickAt(e); if (i !== S.hover) { S.hover = i; updateInfo(i); } }
  });
  el.addEventListener("pointerup", (e) => {
    if (!down) return; const r = el.getBoundingClientRect(), moved = Math.hypot(e.clientX - down.x, e.clientY - down.y); const mode = e.shiftKey ? "add" : e.altKey ? "remove" : "replace";
    if (S.tool === "box") { rect.style.display = "none"; if (moved > 4) boxSelect(down.rx, down.ry, e.clientX - r.left, e.clientY - r.top, mode); else if (mode === "replace") { S.selected = new Set(); applySelection(); emit("replace"); } }
    else if (moved < 5) { const i = pickAt(e);
      if (S.tool === "measure") { if (i >= 0) { S.measure = S.measure.length >= 2 ? [i] : S.measure.concat(i); S.lastDist = undefined; drawMeasure(); updateInfo(i); } }
      else if (i >= 0) { if (mode === "add") S.selected.add(i); else if (mode === "remove") S.selected.delete(i); else S.selected = new Set([i]); applySelection(); emit(mode); }
      else if (mode === "replace" && S.selected.size) { S.selected = new Set(); applySelection(); emit("replace"); } }
    down = null;
  });
}

// ----------------------------------------------------------------------------------- render loop
function resize() {
  const w = $("stage").clientWidth, h = $("stage").clientHeight; if (!w || !h) return; renderer.setSize(w, h, false); const a = w / h;
  persp.aspect = a; persp.updateProjectionMatrix(); const s = 7; ortho.left = -s * a; ortho.right = s * a; ortho.top = s; ortho.bottom = -s; ortho.updateProjectionMatrix();
}
function render() {
  renderer.setScissorTest(false); renderer.setViewport(0, 0, renderer.domElement.width, renderer.domElement.height); renderer.render(scene3, camera);
  const sz = 84, W = renderer.domElement.width, H = renderer.domElement.height; renderer.setScissorTest(true); renderer.setViewport(8, 8, sz, sz); renderer.setScissor(8, 8, sz, sz); renderer.autoClear = false; renderer.clearDepth();
  gizmoCam.position.copy(camera.position).sub(controls.target).setLength(3); gizmoCam.up.copy(camera.up); gizmoCam.lookAt(0, 0, 0); renderer.render(gizmoScene, gizmoCam); renderer.autoClear = true; renderer.setScissorTest(false);
}
let last = performance.now();
function loop(now) {
  requestAnimationFrame(loop); const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (S.playing && loaded) { S.t += S.dir * dt * 0.45; if (S.t >= 1) { S.t = 1; S.dir = -1; } if (S.t <= 0) { S.t = 0; S.dir = 1; } $("t").value = Math.round(S.t * 100); applyPositions(); }
  controls.update(); render();
}
function init() {
  const cv = $("c"); renderer = new THREE.WebGLRenderer({ canvas: cv, antialias: true, preserveDrawingBuffer: true }); renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  scene3 = new THREE.Scene(); scene3.background = new THREE.Color(S.bg); scene3.fog = new THREE.Fog(S.bg, 14, 38);
  persp = new THREE.PerspectiveCamera(42, 1, 0.1, 200); ortho = new THREE.OrthographicCamera(-7, 7, 7, -7, -100, 200); camera = persp; camera.up.set(0, 0, 1); camera.position.set(9, 7, 11);
  controls = new OrbitControls(camera, cv); controls.enableDamping = true; controls.dampingFactor = 0.12; controls.autoRotateSpeed = 1.6;
  scene3.add(new THREE.HemisphereLight(0xffffff, 0xb9c0cc, 1.1)); const dl = new THREE.DirectionalLight(0xffffff, 1.6); dl.position.set(6, -8, 10); scene3.add(dl);
  group = new THREE.Group(); scene3.add(group); raycaster = new THREE.Raycaster(); clip = [];
  const gt = (THEMES[S._theme] || THEMES.light).grid; gridObj = new THREE.GridHelper(14, 14, gt[0], gt[1]); gridObj.rotation.x = Math.PI / 2; gridObj.position.z = -5.4; scene3.add(gridObj);
  gizmoScene = new THREE.Scene(); gizmoCam = new THREE.PerspectiveCamera(40, 1, 0.1, 10); const ax = new THREE.AxesHelper(1); gizmoScene.add(ax);
  new ResizeObserver(resize).observe($("stage")); resize(); wire(); setTool("orbit"); document.querySelector('[data-rep="spheres"]').classList.add("on"); requestAnimationFrame(loop);
}

// ----------------------------------------------------------------------------------- Streamlit protocol
function onRender(args) {
  if (!args) return; if (args.theme && !renderer) { const t0 = THEMES[args.theme] || THEMES.light; S.bg = t0.bg; setTheme(args.theme); }
  if (!renderer) init();
  if (args.theme) setTheme(args.theme);
  if (args.height) { $("app").style.height = args.height + "px"; send("streamlit:setFrameHeight", { height: args.height }); }
  if (args.rep && !S.ready) { S.rep = args.rep; }
  if (args.color && !S.ready) { S.color = args.color; }
  const repChanged = args.rep && args.rep !== S._argRep, colChanged = args.color && args.color !== S._argColor;
  S._argRep = args.rep; S._argColor = args.color;
  if (args.scene && args.scene_id !== S.sceneId) { S.sceneId = args.scene_id; if (!S.ready) { setRep(S.rep); } buildScene(args.scene, !!args.refit || !S.ready); applyColors(); }
  if (S.ready && S.scene) {
    if (repChanged && args.rep !== S.rep) setRep(args.rep);
    if (colChanged && args.color !== S.color && ($("color").querySelector(`option[value="${args.color}"]`) || [...$("color").options].some((o) => o.text === args.color))) { S.color = args.color; $("color").value = args.color; applyColors(); }
  }
  if (args.sel_rev !== S.selRev) { S.selRev = args.sel_rev; S.selected = new Set(args.selected || []); if (S.scene) applySelection(); }
  resize();
}
window.addEventListener("message", (e) => { if (e.data && e.data.type === "streamlit:render") onRender(e.data.args); });
window.__mequaViewer = { S, onRender, boxSelectForTest: (a, b, c, d, m) => boxSelect(a, b, c, d, m || "replace"), project: (i) => { const v = worldOf(i).project(camera); const r = renderer.domElement.getBoundingClientRect(); return [(v.x * 0.5 + 0.5) * r.width, (-v.y * 0.5 + 0.5) * r.height]; } };
send("streamlit:componentReady", { apiVersion: 1 });
