// three.js demo for the *full* bijective projection shell library compiled
// to WASM. Builds a real PrismCage via the SIGGRAPH Asia 2020 pipeline (no
// per-vertex normal stand-in) and exercises the bijective Phong projection
// on query points sampled inside the prismatic shell.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import PrismWASM from "./prism_full_wasm.js";
import { createShellMappingMesh } from "./shellMapping.js";

const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const setStatus = (s) => (statusEl.textContent = s);
const setError = (s) => (errorEl.textContent = s);

const renderer = new THREE.WebGLRenderer({ antialias: true });
// Pixel ratio of 1 keeps the raymarcher's per-fragment cost manageable on
// high-DPI mobile displays. Costs a bit of crispness on desktop, but the
// shader is by far the dominant cost.
renderer.setPixelRatio(1);
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById("app").appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x101418);
scene.fog = new THREE.Fog(0x101418, 6, 18);

const camera = new THREE.PerspectiveCamera(
  45, window.innerWidth / window.innerHeight, 0.05, 100);
camera.position.set(2.6, 1.6, 3.2);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 0.9);
key.position.set(3, 4, 2);
scene.add(key);
const rim = new THREE.DirectionalLight(0x88aaff, 0.4);
rim.position.set(-2, 1, -2);
scene.add(rim);

const root = new THREE.Group();
scene.add(root);

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---------- mesh utilities ----------
function indexAndDedupe(geometry) {
  const pos = geometry.attributes.position.array;
  const idx = geometry.index ? geometry.index.array : null;
  const triCount = idx ? idx.length / 3 : pos.length / 9;
  const map = new Map();
  const V = [];
  const F = new Int32Array(triCount * 3);
  const key = (x, y, z) => `${x.toFixed(6)},${y.toFixed(6)},${z.toFixed(6)}`;
  for (let f = 0; f < triCount; f++) {
    for (let k = 0; k < 3; k++) {
      const vi = idx ? idx[f * 3 + k] : f * 3 + k;
      const x = pos[vi * 3], y = pos[vi * 3 + 1], z = pos[vi * 3 + 2];
      const kk = key(x, y, z);
      let outIdx = map.get(kk);
      if (outIdx === undefined) {
        outIdx = V.length / 3;
        map.set(kk, outIdx);
        V.push(x, y, z);
      }
      F[f * 3 + k] = outIdx;
    }
  }
  return { V: new Float64Array(V), F };
}

function makeMeshGeometry(name) {
  switch (name) {
    case "torusKnot":
      return new THREE.TorusKnotGeometry(0.7, 0.22, 80, 14, 2, 3);
    case "torusKnot32":
      return new THREE.TorusKnotGeometry(0.7, 0.22, 96, 14, 3, 2);
    case "torusKnot37":
      return new THREE.TorusKnotGeometry(0.7, 0.18, 128, 14, 3, 7);
    case "torus":
      return new THREE.TorusGeometry(0.8, 0.28, 14, 60);
    case "sphere":
      return new THREE.IcosahedronGeometry(0.9, 3);
    case "dodecahedron":
      return new THREE.DodecahedronGeometry(0.9, 0);
    case "octahedron":
      return new THREE.OctahedronGeometry(0.95, 2);
    case "capsule":
      return new THREE.CapsuleGeometry(0.5, 0.8, 6, 16);
    case "lathe": {
      // Vase profile via lathe — closed by snapping endpoints to the y-axis.
      const pts = [];
      const N = 24;
      for (let i = 0; i <= N; i++) {
        const t = i / N;
        const y = (t - 0.5) * 1.6;
        const r = i === 0 || i === N ? 0 :
          0.42 + 0.20 * Math.sin(t * Math.PI * 3) +
          0.06 * Math.sin(t * Math.PI * 9);
        pts.push(new THREE.Vector2(r, y));
      }
      return new THREE.LatheGeometry(pts, 36);
    }
    case "bunnyish": {
      const g = new THREE.IcosahedronGeometry(0.9, 4);
      const pos = g.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
        const r = Math.sqrt(x * x + y * y + z * z);
        const lump =
          0.18 * Math.sin(2.1 * x + 1.7 * y) +
          0.10 * Math.cos(3.0 * z + 0.3 * y);
        const s = (r + lump) / r;
        pos.setXYZ(i, x * s, y * s, z * s);
      }
      g.computeVertexNormals();
      return g;
    }
    default:
      return new THREE.TorusKnotGeometry(0.7, 0.22, 80, 14);
  }
}

// ---------- manifold-3d CSG geometries ----------
// manifold-3d is loaded lazily on first use (~700 KB WASM). It guarantees
// the output of every Boolean is a closed manifold mesh, which is exactly
// what PrismCage::construct_cage requires.
let manifoldModulePromise = null;
async function getManifoldModule() {
  if (manifoldModulePromise) return manifoldModulePromise;
  manifoldModulePromise = (async () => {
    const Module = (await import(
      "https://cdn.jsdelivr.net/npm/manifold-3d@3.4.1/manifold.js")).default;
    const m = await Module();
    m.setup();
    m.setCircularSegments(48);  // controls cylinder/sphere tessellation
    return m;
  })();
  return manifoldModulePromise;
}

// Convert a manifold Mesh to (V, F) for buildShell.
function manifoldMeshToVF(mesh) {
  const numProp = mesh.numProp;
  const nV = mesh.numVert;
  const V = new Float64Array(nV * 3);
  for (let i = 0; i < nV; i++) {
    V[3 * i + 0] = mesh.vertProperties[i * numProp + 0];
    V[3 * i + 1] = mesh.vertProperties[i * numProp + 1];
    V[3 * i + 2] = mesh.vertProperties[i * numProp + 2];
  }
  const F = new Int32Array(mesh.triVerts.length);
  F.set(mesh.triVerts);
  return { V, F };
}

// Build a manifold-3d CSG shape and return (V, F). Centers and normalises
// the output so it lands at the origin with a comfortable scale for the
// camera.
async function makeCSGGeometry(name) {
  const M = await getManifoldModule();
  const { Manifold } = M;
  let result;
  const owned = [];
  const own = (m) => { owned.push(m); return m; };
  try {
    switch (name) {
      case "csg-drilled-cube": {
        const cube = own(Manifold.cube([1.6, 1.6, 1.6], true));
        const sphere = own(Manifold.sphere(1.0, 64));
        result = own(cube.subtract(sphere)).getMesh();
        break;
      }
      case "csg-rounded-cube": {
        const cube = own(Manifold.cube([1.4, 1.4, 1.4], true));
        const sphere = own(Manifold.sphere(0.95, 64));
        result = own(cube.intersect(sphere)).getMesh();
        break;
      }
      case "csg-snowman": {
        const a = own(Manifold.sphere(0.7, 48));
        const b = own(own(Manifold.sphere(0.55, 48)).translate([0, 0.95, 0]));
        const c = own(own(Manifold.sphere(0.42, 48)).translate([0, 1.7, 0]));
        result = own(own(a.add(b)).add(c)).getMesh();
        break;
      }
      case "csg-drilled-torus": {
        // manifold doesn't ship a torus primitive, so revolve a circle
        // around y by sampling. Use a tube via two spheres + cylinder
        // is brittle; instead build a torus from a polygon revolution.
        const torus = own(buildTorusManifold(M, 0.85, 0.32, 64, 24));
        const drill = own(
          own(Manifold.cylinder(2.4, 0.18, 0.18, 32, true))
            .rotate([90, 0, 0]));
        result = own(torus.subtract(drill)).getMesh();
        break;
      }
      case "csg-knurl": {
        // A cube with a grid of small cylindrical bores, the classic CAD demo.
        let m = own(Manifold.cube([1.4, 1.4, 1.4], true));
        const drillR = 0.10, drillH = 1.6;
        for (let i = -1; i <= 1; i++) {
          for (let j = -1; j <= 1; j++) {
            const dx = i * 0.40, dz = j * 0.40;
            const cyl = own(
              own(Manifold.cylinder(drillH, drillR, drillR, 24, true))
                .translate([dx, 0, dz]));
            m = own(m.subtract(cyl));
          }
        }
        result = m.getMesh();
        break;
      }
      default:
        throw new Error("unknown CSG geometry: " + name);
    }
    const { V, F } = manifoldMeshToVF(result);
    centerAndScaleInPlace(V, 0.9);
    return { V, F };
  } finally {
    for (const m of owned) m.delete?.();
  }
}

// Build a torus manifold by revolving a polygonal cross-section around y.
function buildTorusManifold(M, R, r, segMajor, segMinor) {
  const pts = [];
  for (let i = 0; i < segMinor; i++) {
    const a = (i / segMinor) * Math.PI * 2;
    pts.push([R + r * Math.cos(a), r * Math.sin(a)]);
  }
  // CrossSection -> revolve
  const cs = M.CrossSection.ofPolygons([pts]);
  return cs.revolve(segMajor);
}

// Center V at origin and scale to fit a unit-ish bounding sphere.
function centerAndScaleInPlace(V, target) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < V.length; i += 3) {
    minX = Math.min(minX, V[i]); maxX = Math.max(maxX, V[i]);
    minY = Math.min(minY, V[i + 1]); maxY = Math.max(maxY, V[i + 1]);
    minZ = Math.min(minZ, V[i + 2]); maxZ = Math.max(maxZ, V[i + 2]);
  }
  const cx = 0.5 * (minX + maxX), cy = 0.5 * (minY + maxY), cz = 0.5 * (minZ + maxZ);
  const half = 0.5 * Math.max(maxX - minX, maxY - minY, maxZ - minZ);
  const s = (target / half) || 1;
  for (let i = 0; i < V.length; i += 3) {
    V[i] = (V[i] - cx) * s;
    V[i + 1] = (V[i + 1] - cy) * s;
    V[i + 2] = (V[i + 2] - cz) * s;
  }
}

// Sample query points inside the shell volume. For each face, randomly pick
// (u, v) on the triangle and t ∈ [0, 1] and lerp between base and top via
// the mid surface. Half the samples land in the base→mid slab, half in mid→top.
function sampleQueriesInShell(baseV, midV, topV, F, nQ) {
  const nF = F.length / 3;
  const samples = new Float64Array(nQ * 3);
  for (let q = 0; q < nQ; q++) {
    const f = (Math.random() * nF) | 0;
    let u = Math.random(), v = Math.random();
    if (u + v > 1) { u = 1 - u; v = 1 - v; }
    const w = 1 - u - v;
    const upperHalf = Math.random() < 0.5;
    const t = Math.random();
    const ia = F[3 * f], ib = F[3 * f + 1], ic = F[3 * f + 2];
    for (let k = 0; k < 3; k++) {
      const a = upperHalf
        ? midV[3 * ia + k] : baseV[3 * ia + k];
      const b = upperHalf
        ? midV[3 * ib + k] : baseV[3 * ib + k];
      const c = upperHalf
        ? midV[3 * ic + k] : baseV[3 * ic + k];
      const a2 = upperHalf
        ? topV[3 * ia + k] : midV[3 * ia + k];
      const b2 = upperHalf
        ? topV[3 * ib + k] : midV[3 * ib + k];
      const c2 = upperHalf
        ? topV[3 * ic + k] : midV[3 * ic + k];
      const lo = a * w + b * u + c * v;
      const hi = a2 * w + b2 * u + c2 * v;
      samples[3 * q + k] = lo * (1 - t) + hi * t;
    }
  }
  return samples;
}

// ---------- three.js builders ----------
function buildSurfaceMesh(V, F, color, opacity = 1.0, thickness = null) {
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(V.length);
  for (let i = 0; i < V.length; i++) pos[i] = V[i];
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setIndex(new THREE.BufferAttribute(new Uint32Array(F.buffer, F.byteOffset, F.length), 1));
  geo.computeVertexNormals();

  let mat;
  if (thickness) {
    // Vertex-color the surface by per-vertex shell thickness — blue = pinched
    // to zero, magenta = full target thickness. Makes the "where did the
    // PrismCage refuse to thicken" pattern obvious.
    let mx = 0;
    for (const t of thickness) if (t > mx) mx = t;
    if (mx <= 0) mx = 1;
    const colors = new Float32Array(V.length);
    for (let i = 0; i < V.length / 3; i++) {
      const t = Math.min(thickness[i] / mx, 1);
      // turbo-ish ramp: blue (cold) → cyan → green → yellow → red (hot)
      const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 3)));
      const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 2)));
      const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * t - 1)));
      colors[3 * i] = r;
      colors[3 * i + 1] = g;
      colors[3 * i + 2] = b;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    mat = new THREE.MeshStandardMaterial({
      vertexColors: true, metalness: 0.05, roughness: 0.55,
      transparent: opacity < 1, opacity, side: THREE.DoubleSide,
    });
  } else {
    mat = new THREE.MeshStandardMaterial({
      color, metalness: 0.05, roughness: 0.55, flatShading: false,
      transparent: opacity < 1, opacity, side: THREE.DoubleSide,
    });
  }
  return new THREE.Mesh(geo, mat);
}

function buildWireMesh(V, F, color, opacity = 0.35) {
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(V.length);
  for (let i = 0; i < V.length; i++) pos[i] = V[i];
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setIndex(new THREE.BufferAttribute(new Uint32Array(F.buffer, F.byteOffset, F.length), 1));
  const mat = new THREE.MeshBasicMaterial({
    color, wireframe: true, transparent: opacity < 1, opacity,
  });
  return new THREE.Mesh(geo, mat);
}

function buildPointCloud(P, color, size = 0.018) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(P), 3));
  const mat = new THREE.PointsMaterial({ color, size, sizeAttenuation: true });
  return new THREE.Points(geo, mat);
}

function buildLineSegments(qP, imgP, hit, color) {
  const segs = [];
  for (let i = 0; i < hit.length; i++) {
    if (!hit[i]) continue;
    segs.push(qP[3 * i], qP[3 * i + 1], qP[3 * i + 2],
              imgP[3 * i], imgP[3 * i + 1], imgP[3 * i + 2]);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(segs), 3));
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 });
  return new THREE.LineSegments(geo, mat);
}

// ---------- main rebuild loop ----------
let Module = null;

function clearGroup(g) {
  while (g.children.length) {
    const c = g.children.pop();
    c.geometry?.dispose?.();
    c.material?.dispose?.();
  }
}

function vecToFloat64(v) {
  const n = v.size();
  const a = new Float64Array(n);
  for (let i = 0; i < n; i++) a[i] = v.get(i);
  return a;
}
function vecToInt32(v) {
  const n = v.size();
  const a = new Int32Array(n);
  for (let i = 0; i < n; i++) a[i] = v.get(i);
  return a;
}
function deleteVecs(obj) {
  for (const k of Object.keys(obj)) {
    if (obj[k] && typeof obj[k].delete === "function") obj[k].delete();
  }
}

// Wipe `root` and add fresh primitives for the current shell snapshot. Called
// after every iteration of remesh_schedule so the user sees the cage inflate.
function renderShell(midV, baseV, topV, Fout, thicknessArr, queries) {
  clearGroup(root);
  const showShell = document.getElementById("showShell").checked;
  const showSurface = document.getElementById("showSurface").checked;
  const showLines = document.getElementById("showLines").checked;
  const showQueries = document.getElementById("showQueries").checked;
  const showShellMap = document.getElementById("showShellMap").checked;

  if (showSurface) {
    root.add(buildSurfaceMesh(midV, Fout, 0x7aa9ff, 0.85, thicknessArr));
  }
  if (showShell) {
    root.add(buildWireMesh(baseV, Fout, 0x22aa55, 1.0));
    root.add(buildWireMesh(topV, Fout, 0xdd3344, 1.0));
  }
  if (showShellMap) {
    const Fbuf = new Int32Array(Fout);
    const pattern = parseInt(document.getElementById("smPattern").value, 10);
    const scale = parseFloat(document.getElementById("smScale").value);
    const bump = parseFloat(document.getElementById("smBump").value);
    // Heightmap raymarcher displaces from mid outward into the upper slab.
    // One mesh suffices — the lower slab is covered by the regular surface
    // / shell wireframe layers if the user wants them.
    root.add(createShellMappingMesh(baseV, midV, topV, Fbuf, {
      pattern, patternScale: scale, bumpHeight: bump,
    }));
  }
  if (queries && showQueries) {
    const { Q, imgP, hit } = queries;
    root.add(buildPointCloud(Q, 0xffc04d, 0.020));
    const imgFiltered = [];
    for (let i = 0; i < hit.length; i++) {
      if (hit[i]) imgFiltered.push(imgP[3 * i], imgP[3 * i + 1], imgP[3 * i + 2]);
    }
    root.add(buildPointCloud(imgFiltered, 0x5be0a4, 0.022));
    if (showLines) {
      root.add(buildLineSegments(Q, imgP, hit, 0xa78bfa));
    }
  }
}

async function rebuild() {
  if (!Module) return;
  const t0 = performance.now();
  clearGroup(root);

  const geomName = document.getElementById("geom").value;
  const thickness = parseFloat(document.getElementById("thick").value);
  const nQ = parseInt(document.getElementById("nq").value, 10);

  let V, F;
  if (geomName.startsWith("csg-")) {
    setStatus("loading manifold-3d…");
    try {
      ({ V, F } = await makeCSGGeometry(geomName));
    } catch (e) {
      setError("CSG build failed: " + (e?.message || e));
      return;
    }
  } else {
    const geom = makeMeshGeometry(geomName);
    ({ V, F } = indexAndDedupe(geom));
  }
  window.__VF = { V, F };

  // Normalise every input mesh to bbox half-extent 0.9 so the thickness
  // slider always means the same thing regardless of source primitive scale.
  centerAndScaleInPlace(V, 0.9);

  // ---- Stage 1: faithful-to-paper construction. ----
  // PrismCage::PrismCage builds an initial cage with extrusion ~ initial_step
  // (paper uses 1e-4 of unit bbox). The cage is uniformly thin at this point
  // — thick shells are produced by Stage 2 (remesh_schedule).
  const INIT_STEP = 1e-4;
  const DOO_EPS = 0.2;
  let shell;
  try {
    shell = Module.buildShell(Array.from(V), Array.from(F), DOO_EPS, INIT_STEP);
  } catch (e) {
    setError("buildShell threw: " + (e?.message || e));
    return;
  }
  let midV = vecToFloat64(shell.midV);
  let baseV = vecToFloat64(shell.baseV);
  let topV = vecToFloat64(shell.topV);
  let Fout = vecToInt32(shell.F);
  let thicknessArr = shell.thickness ? vecToFloat64(shell.thickness) : null;
  deleteVecs(shell);

  if (!Fout.length) {
    setStatus(`${geomName}: PrismCage construction returned empty mesh.`);
    return;
  }

  // ---- Stage 2: remesh_schedule grows the cage toward target thickness. ----
  // Mirror src/construct_shell.cpp::remesh_schedule but cap iterations so the
  // demo stays interactive. Re-render between steps so the user sees the
  // shell inflate.
  const targetEdge = (() => {
    let lo = [Infinity, Infinity, Infinity];
    let hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < V.length; i += 3) {
      for (let k = 0; k < 3; k++) {
        if (V[i + k] < lo[k]) lo[k] = V[i + k];
        if (V[i + k] > hi[k]) hi[k] = V[i + k];
      }
    }
    const bb = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    return bb * 0.2;  // breve_bin's default --edge ratio
  })();

  Module.setRemeshOptions(thickness, targetEdge, 0.1);

  // First render: thin initial cage. Lets the user see Stage 1 before Stage 2
  // grows it.
  renderShell(midV, baseV, topV, Fout, thicknessArr, /*queries*/ null);

  // Stage 2: per-pass schedule. We call collapseStep / flipStep / smoothStep
  // individually and re-render between each so the user sees the cage
  // simplify (collapse) and inflate (smooth). Mirrors the schedule in
  // src/construct_shell.cpp:remesh_schedule but exposes the per-pass
  // boundaries to JS for progress tracking.
  const maxIters = parseInt(document.getElementById("growIters").value, 10);
  let iters = 0;
  let collapseTotal = 0;
  const passTotals = { collapse_ms: 0, flip_ms: 0, smooth_ms: 0 };
  const refreshAfterPass = async (label, passInfo) => {
    const s = Module.getShell();
    midV = vecToFloat64(s.midV);
    baseV = vecToFloat64(s.baseV);
    topV = vecToFloat64(s.topV);
    Fout = vecToInt32(s.F);
    thicknessArr = vecToFloat64(s.thickness);
    deleteVecs(s);
    renderShell(midV, baseV, topV, Fout, thicknessArr, /*queries*/ null);
    const meanT = thicknessArr.reduce((a, b) => a + b, 0) / thicknessArr.length;
    setStatus(`${geomName} · iter ${iters + 1}/${maxIters} ${label} · ` +
              `cage V=${midV.length / 3} F=${Fout.length / 3} · ` +
              `mean thick=${meanT.toFixed(4)} · ` +
              `pass=${passInfo} · ` +
              `cum collapse=${passTotals.collapse_ms.toFixed(0)}ms ` +
              `flip=${passTotals.flip_ms.toFixed(0)}ms ` +
              `smooth=${passTotals.smooth_ms.toFixed(0)}ms`);
    await new Promise(r => requestAnimationFrame(r));
  };

  outer: for (iters = 0; iters < maxIters; iters++) {
    // 1) collapse
    const cRes = Module.collapseStep();
    passTotals.collapse_ms += cRes.ms;
    collapseTotal += cRes.count;
    await refreshAfterPass("collapse", `${cRes.count} collapses, ${cRes.ms.toFixed(0)} ms`);

    // 2) flip + smooth, twice (matches construct_shell.cpp:172-178)
    for (let j = 0; j < 2; j++) {
      const fRes = Module.flipStep();
      passTotals.flip_ms += fRes.ms;
      await refreshAfterPass(`flip${j ? "₂" : "₁"}`, `${fRes.ms.toFixed(0)} ms`);
      const sRes = Module.smoothStep();
      passTotals.smooth_ms += sRes.ms;
      await refreshAfterPass(`smooth${j ? "₂" : "₁"}`, `${sRes.ms.toFixed(0)} ms`);
    }

    // Convergence early-out: same rule as construct_shell.cpp:185 — break
    // when the collapse pass found ≤ 0.01% of the cage's vertex count.
    const nVCur = midV.length / 3;
    if (cRes.count <= 1e-4 * nVCur) break outer;
  }

  // Polish loop: 20 trailing flip+smooth iterations (no collapse). This is
  // where most of the thickness growth happens for already-uniform meshes,
  // matching the second loop in construct_shell.cpp:191-200.
  let prevMean = 0;
  for (let i = 0; i < 20; i++) {
    const fRes = Module.flipStep();
    passTotals.flip_ms += fRes.ms;
    const sRes = Module.smoothStep();
    passTotals.smooth_ms += sRes.ms;

    const s = Module.getShell();
    midV = vecToFloat64(s.midV);
    baseV = vecToFloat64(s.baseV);
    topV = vecToFloat64(s.topV);
    Fout = vecToInt32(s.F);
    thicknessArr = vecToFloat64(s.thickness);
    deleteVecs(s);
    renderShell(midV, baseV, topV, Fout, thicknessArr, /*queries*/ null);
    const meanT = thicknessArr.reduce((a, b) => a + b, 0) / thicknessArr.length;
    setStatus(`${geomName} · polish ${i + 1}/20 · cage V=${midV.length / 3} · ` +
              `mean thick=${meanT.toFixed(4)} · ` +
              `pass=${(fRes.ms + sRes.ms).toFixed(0)} ms · ` +
              `cum collapse=${passTotals.collapse_ms.toFixed(0)}ms ` +
              `flip=${passTotals.flip_ms.toFixed(0)}ms ` +
              `smooth=${passTotals.smooth_ms.toFixed(0)}ms`);
    await new Promise(r => requestAnimationFrame(r));

    // Plateau early-out: stop when the mean isn't improving by >1%.
    if (i > 0 && meanT < prevMean * 1.01) break;
    prevMean = meanT;
  }

  // ---- Stage 3: project queries through the final shell. ----
  const Q = sampleQueriesInShell(baseV, midV, topV, Fout, nQ);
  let proj;
  try {
    proj = Module.projectPoints(Array.from(Q));
  } catch (e) {
    setError("projectPoints threw: " + (e?.message || e));
    return;
  }
  const imgP = vecToFloat64(proj.imageP);
  const hit = vecToInt32(proj.hit);
  const stratum = vecToInt32(proj.stratum);
  deleteVecs(proj);

  renderShell(midV, baseV, topV, Fout, thicknessArr, { Q, imgP, hit });

  const hits = hit.reduce((a, b) => a + b, 0);
  const lower = stratum.reduce((a, s) => a + (s === 0 ? 1 : 0), 0);
  const upper = stratum.reduce((a, s) => a + (s === 1 ? 1 : 0), 0);
  const dt = performance.now() - t0;
  let thickStats = "";
  if (thicknessArr && thicknessArr.length) {
    let tmin = Infinity, tmax = 0, tsum = 0, zeros = 0;
    for (const t of thicknessArr) {
      if (t < tmin) tmin = t; if (t > tmax) tmax = t; tsum += t;
      if (t < 1e-6) zeros++;
    }
    const tmean = tsum / thicknessArr.length;
    thickStats = ` · thickness min=${tmin.toExponential(2)} mean=${tmean.toFixed(4)} max=${tmax.toFixed(4)} zero=${zeros}`;
  }
  setStatus(
    `${geomName} · input V=${V.length / 3} F=${F.length / 3} · ` +
    `cage V=${midV.length / 3} F=${Fout.length / 3} · ` +
    `iters=${iters} collapses=${collapseTotal} · ` +
    `queries=${nQ} hit=${hits} (lower=${lower}, upper=${upper}) · ` +
    `${dt.toFixed(0)} ms` + thickStats);
  setError("");
}

function bindUI() {
  document.getElementById("geom").addEventListener("change", rebuild);
  document.getElementById("thick").addEventListener("input", (e) => {
    document.getElementById("thickVal").textContent = ` (${(+e.target.value).toFixed(3)})`;
    scheduleRebuild();
  });
  document.getElementById("growIters").addEventListener("input", (e) => {
    document.getElementById("growVal").textContent = ` (${e.target.value})`;
    scheduleRebuild();
  });
  document.getElementById("nq").addEventListener("input", (e) => {
    document.getElementById("nqVal").textContent = ` (${e.target.value})`;
    scheduleRebuild();
  });
  for (const id of ["showShell", "showSurface", "showLines", "showQueries",
                    "showShellMap", "smPattern"]) {
    document.getElementById(id).addEventListener("change", rebuild);
  }
  document.getElementById("smScale").addEventListener("input", rebuild);
  document.getElementById("smBump").addEventListener("input", rebuild);
  document.getElementById("reseed").addEventListener("click", rebuild);
}

let _rebuildPending = null;
function scheduleRebuild() {
  cancelAnimationFrame(_rebuildPending);
  _rebuildPending = requestAnimationFrame(rebuild);
}

PrismWASM().then((mod) => {
  Module = mod;
  window.__M = mod;
  setStatus("WASM ready (real PrismCage build).");
  bindUI();
  document.getElementById("thickVal").textContent =
    ` (${(+document.getElementById("thick").value).toFixed(3)})`;
  document.getElementById("growVal").textContent =
    ` (${document.getElementById("growIters").value})`;
  document.getElementById("nqVal").textContent =
    ` (${document.getElementById("nq").value})`;
  rebuild();
}).catch((e) => {
  setError("WASM load failed: " + (e?.message || e));
});

function tick() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();
