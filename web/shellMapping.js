// Shell-mapping heightmap raymarcher with cross-prism neighbour walk.
//
// For each PrismCage face, render the prism's top triangle. The fragment
// shader raymarches from the entry point (canonical t = 1) inward through
// the slab. At each step we decompose the world position into canonical
// (u, v, t) using the same Phong projection as src/prism/phong/projection.cpp.
// When the sample exits the current prism, we look up the adjacent prism
// across the shared edge and continue marching there. This mirrors what
// the paper calls the "section-traversal" of the shell — a ray's image
// under the inverse projection is a piecewise-linear polyline that crosses
// from one prism's tetrahedral decomposition into the next.
//
// Per-prism data (corners, split way, 3 side-neighbour indices) lives in a
// DataTexture so the fragment shader can dereference any prism on demand.
// Per-instance the geometry only carries prismIdx; the vertex positions
// are baked from the actual top-triangle coordinates so the VS just runs
// the standard model/view/projection.

import * as THREE from "three";

const VERTEX_SHADER = /* glsl */`
attribute float prismIdx;
varying vec3 vWorldPos;
varying float vPrismIdx;

void main() {
  vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
  vPrismIdx = prismIdx;
  gl_Position = projectionMatrix * viewMatrix * vec4(vWorldPos, 1.0);
}
`;

const FRAGMENT_SHADER = /* glsl */`
precision highp float;
precision highp sampler2D;

varying vec3 vWorldPos;
varying float vPrismIdx;

uniform sampler2D uPrismData;     // width = nPrisms, height = 7
uniform float uNumPrisms;
uniform float uPatternScale;
uniform float uBumpHeight;
uniform int uPattern;
uniform float uStepSize;
uniform int uMaxSteps;
uniform int uMaxPrismHops;
uniform vec3 uTintLow;
uniform vec3 uTintHigh;
uniform vec3 uLightDir;

// Texel fetch helpers. Each prism's data lives in a column of width 1, with
// 7 rows:
//   0..2  = base corners (mid for the upper slab)  rgb = xyz
//   3..5  = top corners                              rgb = xyz
//   6     = (splitWay, neighbour01, neighbour12, neighbour20) in rgba.
//   neighbour i is the prism across the side opposite local corner i,
//   i.e. the side connecting corners i+1 and i+2 (mod 3) on the bottom
//   triangle. -1 (encoded as a negative float) means boundary.
vec4 prismRow(float prismIdx, float row) {
  float u = (prismIdx + 0.5) / uNumPrisms;
  float v = (row + 0.5) / 7.0;
  return texture2D(uPrismData, vec2(u, v));
}

vec3 prismCorner(float prismIdx, int idx) {
  return prismRow(prismIdx, float(idx)).rgb;
}

float prismSplitWay(float prismIdx) {
  return prismRow(prismIdx, 6.0).r;
}

vec3 prismNeighbors(float prismIdx) {
  // Returns 3 neighbour prism indices (or -1 sentinel) packed in gba.
  return prismRow(prismIdx, 6.0).gba;
}

float orient3d(vec3 a, vec3 b, vec3 c, vec3 d) {
  return determinant(mat3(b - a, c - a, d - a));
}

bool point_in_tet(vec3 p, vec3 T0, vec3 T1, vec3 T2, vec3 T3) {
  const float EPS = -1e-7;
  return orient3d(T0, T3, T1, p) >= EPS
      && orient3d(T1, T3, T2, p) >= EPS
      && orient3d(T0, T1, T2, p) >= EPS
      && orient3d(T0, T2, T3, p) >= EPS;
}

vec4 baryTet(vec3 p, vec3 a, vec3 b, vec3 c, vec3 d) {
  float vol = orient3d(a, b, c, d);
  if (abs(vol) < 1e-20) return vec4(0.25);
  return vec4(
    orient3d(p, b, c, d),
    orient3d(a, p, c, d),
    orient3d(a, b, p, d),
    orient3d(a, b, c, p)
  ) / vol;
}

const vec3 CAN0 = vec3(0.0, 0.0, 0.0);
const vec3 CAN1 = vec3(1.0, 0.0, 0.0);
const vec3 CAN2 = vec3(0.0, 1.0, 0.0);
const vec3 CAN3 = vec3(0.0, 0.0, 1.0);
const vec3 CAN4 = vec3(1.0, 0.0, 1.0);
const vec3 CAN5 = vec3(0.0, 1.0, 1.0);

vec3 canonicalCorner(int i) {
  return i == 0 ? CAN0 :
         i == 1 ? CAN1 :
         i == 2 ? CAN2 :
         i == 3 ? CAN3 :
         i == 4 ? CAN4 : CAN5;
}

// decomposePrism for an arbitrary prism, fetching corners from uPrismData.
bool decomposeAny(float prismIdx, vec3 p, out vec3 uvt) {
  vec3 V0 = prismCorner(prismIdx, 0);
  vec3 V1 = prismCorner(prismIdx, 1);
  vec3 V2 = prismCorner(prismIdx, 2);
  vec3 V3 = prismCorner(prismIdx, 3);
  vec3 V4 = prismCorner(prismIdx, 4);
  vec3 V5 = prismCorner(prismIdx, 5);
  float splitWay = prismSplitWay(prismIdx);

  // TETRA_SPLIT_A = {{0,3,4,5}, {1,4,2,0}, {2,5,0,4}}
  // TETRA_SPLIT_B = {{0,3,4,5}, {1,4,5,0}, {2,5,0,1}}
  for (int i = 0; i < 3; ++i) {
    ivec4 t;
    if (splitWay > 0.5) {
      t = i == 0 ? ivec4(0,3,4,5) :
          i == 1 ? ivec4(1,4,2,0) :
                   ivec4(2,5,0,4);
    } else {
      t = i == 0 ? ivec4(0,3,4,5) :
          i == 1 ? ivec4(1,4,5,0) :
                   ivec4(2,5,0,1);
    }
    vec3 T0 =
        t.x == 0 ? V0 : t.x == 1 ? V1 : t.x == 2 ? V2 :
        t.x == 3 ? V3 : t.x == 4 ? V4 : V5;
    vec3 T1 =
        t.y == 0 ? V0 : t.y == 1 ? V1 : t.y == 2 ? V2 :
        t.y == 3 ? V3 : t.y == 4 ? V4 : V5;
    vec3 T2 =
        t.z == 0 ? V0 : t.z == 1 ? V1 : t.z == 2 ? V2 :
        t.z == 3 ? V3 : t.z == 4 ? V4 : V5;
    vec3 T3 =
        t.w == 0 ? V0 : t.w == 1 ? V1 : t.w == 2 ? V2 :
        t.w == 3 ? V3 : t.w == 4 ? V4 : V5;

    if (point_in_tet(p, T0, T1, T2, T3)) {
      vec4 b = baryTet(p, T0, T1, T2, T3);
      uvt = b.x * canonicalCorner(t.x) + b.y * canonicalCorner(t.y) +
            b.z * canonicalCorner(t.z) + b.w * canonicalCorner(t.w);
      return true;
    }
  }
  return false;
}

// ---- 2D heightfield ----
float hash21(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash21(i + vec2(0,0)), hash21(i + vec2(1,0)), u.x),
             mix(hash21(i + vec2(0,1)), hash21(i + vec2(1,1)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 4; ++i) {
    v += a * vnoise(p);
    p *= 2.0; a *= 0.5;
  }
  return v;
}

float hexBumps2D(vec2 uv) {
  vec2 s = uv * vec2(uPatternScale * 1.732, uPatternScale);
  vec2 h1 = vec2(s.x, s.y);
  vec2 h2 = vec2(s.x + uPatternScale * 0.866, s.y + 0.5);
  vec2 c1 = h1 - (floor(h1) + 0.5);
  vec2 c2 = h2 - (floor(h2) + 0.5);
  return smoothstep(0.5, 0.15, min(length(c1), length(c2)));
}

float brickBumps2D(vec2 uv) {
  float row = floor(uv.y * uPatternScale);
  vec2 b = vec2(uv.x * uPatternScale + 0.5 * mod(row, 2.0),
                uv.y * uPatternScale);
  vec2 c = abs(fract(b) - 0.5);
  return smoothstep(0.0, 0.08, 0.5 - max(c.x, c.y));
}

float pattern2D(vec2 uv) {
  if (uPattern == 0) return hexBumps2D(uv);
  if (uPattern == 1) return brickBumps2D(uv);
  return fbm(uv * uPatternScale);
}

float triplanar(vec3 p, vec3 n) {
  vec3 w = abs(normalize(n));
  w = pow(w, vec3(4.0));
  w /= (w.x + w.y + w.z + 1e-6);
  return w.x * pattern2D(p.yz)
       + w.y * pattern2D(p.zx)
       + w.z * pattern2D(p.xy);
}

float heightmap(vec3 worldP, vec3 nWorld) {
  return triplanar(worldP, nWorld) * uBumpHeight;
}

// Mid-surface image of a (u, v) inside the upper slab. mid corners are
// stored in rows 0..2, top corners in 3..5.
vec3 midImage(float prismIdx, vec3 uvt) {
  return (1.0 - uvt.x - uvt.y) * prismCorner(prismIdx, 0)
       + uvt.x                 * prismCorner(prismIdx, 1)
       + uvt.y                 * prismCorner(prismIdx, 2);
}
vec3 prismPillar(float prismIdx) {
  return normalize(prismCorner(prismIdx, 3) - prismCorner(prismIdx, 0));
}

// Try the 3 side neighbours of prismIdx and pick whichever decomposes
// the world point p.  Returns -1.0 if none does (i.e. ray has truly left
// the shell volume).
float findContainingNeighbor(float prismIdx, vec3 p, out vec3 uvt) {
  vec3 nbrs = prismNeighbors(prismIdx);
  for (int k = 0; k < 3; ++k) {
    float nbr = k == 0 ? nbrs.x : k == 1 ? nbrs.y : nbrs.z;
    if (nbr < 0.0) continue;
    if (decomposeAny(nbr, p, uvt)) return nbr;
  }
  return -1.0;
}

void main() {
  vec3 ro = vWorldPos;
  vec3 rd = normalize(vWorldPos - cameraPosition);

  float currentPrism = vPrismIdx;
  vec3 nSurface = prismPillar(currentPrism);

  bool everInside = false;
  int hops = 0;
  for (int i = 0; i < 512; ++i) {
    if (i >= uMaxSteps) break;
    vec3 p = ro + rd * (float(i) * uStepSize);

    vec3 uvt;
    if (!decomposeAny(currentPrism, p, uvt)) {
      // Try walking to a side neighbour.
      if (hops < uMaxPrismHops) {
        float nb = findContainingNeighbor(currentPrism, p, uvt);
        if (nb >= 0.0) {
          currentPrism = nb;
          nSurface = prismPillar(currentPrism);
          ++hops;
          // Re-decompose once (already done by findContainingNeighbor) and
          // fall through to the heightmap test below.
        } else {
          break;
        }
      } else {
        break;
      }
    }
    everInside = true;

    vec3 imgP = midImage(currentPrism, uvt);
    float h = heightmap(imgP, nSurface);
    if (uvt.z <= h) {
      const float dx = 0.005;
      float hu = heightmap(midImage(currentPrism, uvt + vec3(dx, 0, 0)), nSurface);
      float hv = heightmap(midImage(currentPrism, uvt + vec3(0, dx, 0)), nSurface);
      vec3 n_canon = normalize(vec3(-(hu - h) / dx, -(hv - h) / dx, 1.0));
      vec3 du = normalize(prismCorner(currentPrism, 1) - prismCorner(currentPrism, 0));
      vec3 dv = normalize(prismCorner(currentPrism, 2) - prismCorner(currentPrism, 0));
      vec3 n_world = normalize(n_canon.x * du + n_canon.y * dv + n_canon.z * nSurface);

      float diffuse = max(0.15, dot(n_world, normalize(uLightDir)));
      vec3 col = mix(uTintLow, uTintHigh,
                     smoothstep(0.0, 1.0, h / max(uBumpHeight, 1e-6)));
      gl_FragColor = vec4(col * diffuse, 1.0);
      return;
    }
  }

  if (!everInside) discard;
  float diffuse = max(0.15, dot(nSurface, normalize(uLightDir)));
  gl_FragColor = vec4(mix(uTintLow, uTintHigh, 0.4) * diffuse, 1.0);
}
`;

// Compute the side-neighbour map: for each face f and each of its 3 edges
// (opposite local corner i), find the other face sharing that edge.
// Returns Float32Array of length 3*nFaces; -1 for boundary.
function computeNeighborMap(F) {
  const nFaces = F.length / 3;
  const neighbors = new Float32Array(nFaces * 3);
  neighbors.fill(-1);
  // Edge key (a, b) with a < b -> [face, localOppositeIdx]
  const edgeMap = new Map();
  for (let f = 0; f < nFaces; ++f) {
    const a = F[3 * f], b = F[3 * f + 1], c = F[3 * f + 2];
    const verts = [a, b, c];
    // Edge i (opposite corner i): (verts[(i+1)%3], verts[(i+2)%3])
    for (let i = 0; i < 3; ++i) {
      const u = verts[(i + 1) % 3];
      const v = verts[(i + 2) % 3];
      const key = u < v ? `${u}_${v}` : `${v}_${u}`;
      const entry = edgeMap.get(key);
      if (entry === undefined) {
        edgeMap.set(key, [f, i]);
      } else {
        const [g, j] = entry;
        neighbors[3 * f + i] = g;
        neighbors[3 * g + j] = f;
        edgeMap.delete(key);
      }
    }
  }
  return neighbors;
}

export function createShellMappingMesh(baseV, midV, topV, F, options = {}) {
  const opts = Object.assign({
    pattern: 0,
    patternScale: 8.0,
    bumpHeight: 0.7,
    stepsPerSlab: 64,
    maxSteps: 256,
    maxPrismHops: 8,
    tintLow: new THREE.Color(0x2a4060),
    tintHigh: new THREE.Color(0xffd28a),
    lightDir: new THREE.Vector3(0.5, 0.7, 0.5),
  }, options);

  const numPrisms = F.length / 3;

  // ---- Per-prism data texture (RGBA float, height = 7, width = nPrisms) ----
  // Rows 0..5 = mid/top corners; row 6 = (splitWay, n01, n12, n20).
  const data = new Float32Array(numPrisms * 7 * 4);
  const setRow = (p, row, r, g, b, a = 0) => {
    const base = (row * numPrisms + p) * 4;
    data[base + 0] = r;
    data[base + 1] = g;
    data[base + 2] = b;
    data[base + 3] = a;
  };

  const neighbors = computeNeighborMap(F);

  // Pre-bake top-triangle world-space positions so the VS doesn't need any
  // texture lookup. Each prism contributes 3 verts (its top triangle).
  const positions = new Float32Array(numPrisms * 3 * 3);
  const prismIdx = new Float32Array(numPrisms * 3);

  for (let f = 0; f < numPrisms; f++) {
    const a = F[3 * f], b = F[3 * f + 1], cc = F[3 * f + 2];
    // Mid corners (rows 0..2)
    setRow(f, 0, midV[3*a],  midV[3*a+1],  midV[3*a+2]);
    setRow(f, 1, midV[3*b],  midV[3*b+1],  midV[3*b+2]);
    setRow(f, 2, midV[3*cc], midV[3*cc+1], midV[3*cc+2]);
    // Top corners (rows 3..5)
    setRow(f, 3, topV[3*a],  topV[3*a+1],  topV[3*a+2]);
    setRow(f, 4, topV[3*b],  topV[3*b+1],  topV[3*b+2]);
    setRow(f, 5, topV[3*cc], topV[3*cc+1], topV[3*cc+2]);
    // Meta (row 6): splitWay + 3 neighbour indices.
    setRow(f, 6,
           (b > cc) ? 1.0 : 0.0,
           neighbors[3 * f + 0],
           neighbors[3 * f + 1],
           neighbors[3 * f + 2]);

    // Top-triangle vertices for this instance.
    for (let k = 0; k < 3; k++) {
      const v = k === 0 ? a : k === 1 ? b : cc;
      positions[(f * 3 + k) * 3 + 0] = topV[3 * v];
      positions[(f * 3 + k) * 3 + 1] = topV[3 * v + 1];
      positions[(f * 3 + k) * 3 + 2] = topV[3 * v + 2];
      prismIdx[f * 3 + k] = f;
    }
  }

  const prismDataTex = new THREE.DataTexture(
      data, numPrisms, 7, THREE.RGBAFormat, THREE.FloatType);
  prismDataTex.minFilter = THREE.NearestFilter;
  prismDataTex.magFilter = THREE.NearestFilter;
  prismDataTex.needsUpdate = true;

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("prismIdx", new THREE.BufferAttribute(prismIdx, 1));

  let minX=Infinity, minY=Infinity, minZ=Infinity;
  let maxX=-Infinity, maxY=-Infinity, maxZ=-Infinity;
  const pillarLens = [];
  for (let i = 0; i < midV.length; i += 3) {
    minX = Math.min(minX, baseV[i], topV[i]);
    minY = Math.min(minY, baseV[i+1], topV[i+1]);
    minZ = Math.min(minZ, baseV[i+2], topV[i+2]);
    maxX = Math.max(maxX, baseV[i], topV[i]);
    maxY = Math.max(maxY, baseV[i+1], topV[i+1]);
    maxZ = Math.max(maxZ, baseV[i+2], topV[i+2]);
    const dx = topV[i] - midV[i];
    const dy = topV[i+1] - midV[i+1];
    const dz = topV[i+2] - midV[i+2];
    pillarLens.push(Math.sqrt(dx*dx + dy*dy + dz*dz));
  }
  pillarLens.sort((a, b) => a - b);
  const medianPillar = pillarLens[Math.floor(pillarLens.length / 2)] || 0.05;
  const stepSize = Math.max(medianPillar / opts.stepsPerSlab, 1e-4);

  geom.boundingBox = new THREE.Box3(
      new THREE.Vector3(minX, minY, minZ),
      new THREE.Vector3(maxX, maxY, maxZ));
  geom.boundingSphere = geom.boundingBox.getBoundingSphere(new THREE.Sphere());

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uPrismData: { value: prismDataTex },
      uNumPrisms: { value: numPrisms },
      uPatternScale: { value: opts.patternScale },
      uBumpHeight: { value: opts.bumpHeight },
      uPattern: { value: opts.pattern },
      uStepSize: { value: stepSize },
      uMaxSteps: { value: opts.maxSteps },
      uMaxPrismHops: { value: opts.maxPrismHops },
      uTintLow: { value: new THREE.Vector3(opts.tintLow.r, opts.tintLow.g, opts.tintLow.b) },
      uTintHigh: { value: new THREE.Vector3(opts.tintHigh.r, opts.tintHigh.g, opts.tintHigh.b) },
      uLightDir: { value: opts.lightDir.clone() },
    },
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    side: THREE.FrontSide,
  });

  const mesh = new THREE.Mesh(geom, material);
  mesh.userData.dispose = () => prismDataTex.dispose();
  return mesh;
}
