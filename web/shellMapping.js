// Shell-mapping raymarcher for the bijective prismatic shell.
//
// For each face of the shell, render the prism's outer boundary as 8
// triangles. In the fragment shader, the world position of each fragment
// is the ray's *entry* into the prism. We raymarch inward along the view
// direction, decompose each sample's world position into canonical-prism
// coordinates (u, v, t) via the same Phong projection the C++ code uses,
// and look up a procedural 3D pattern at (u, v, t).
//
// This is a port of Porumbescu et al. 2005 "Shell maps" using PrismCage's
// triangulated-prism parametrisation. The bijection (u, v, t) <-> world
// position is exact by construction, so the pattern follows the surface
// curvature without distortion.

import * as THREE from "three";

// Indices into the 6-corner array V = [base[a], base[b], base[c],
// top[a], top[b], top[c]]. Triangles wind so the normal points outward.
const PRISM_TRIS = [
  // bottom (base) — outward = -t direction
  [0, 2, 1],
  // top — outward = +t direction
  [3, 4, 5],
  // 3 side rectangles, each split into 2 triangles
  [0, 1, 4], [0, 4, 3],
  [1, 2, 5], [1, 5, 4],
  [2, 0, 3], [2, 3, 5],
];

const VERTEX_SHADER = /* glsl */`
attribute float cornerIdx;
attribute vec3 c0;
attribute vec3 c1;
attribute vec3 c2;
attribute vec3 c3;
attribute vec3 c4;
attribute vec3 c5;
attribute float splitWay;

varying vec3 vWorldPos;
varying vec3 vC0;
varying vec3 vC1;
varying vec3 vC2;
varying vec3 vC3;
varying vec3 vC4;
varying vec3 vC5;
varying float vSplitWay;

void main() {
  int ci = int(cornerIdx + 0.5);
  vec3 pos =
      ci == 0 ? c0 :
      ci == 1 ? c1 :
      ci == 2 ? c2 :
      ci == 3 ? c3 :
      ci == 4 ? c4 : c5;
  vWorldPos = (modelMatrix * vec4(pos, 1.0)).xyz;
  vC0 = (modelMatrix * vec4(c0, 1.0)).xyz;
  vC1 = (modelMatrix * vec4(c1, 1.0)).xyz;
  vC2 = (modelMatrix * vec4(c2, 1.0)).xyz;
  vC3 = (modelMatrix * vec4(c3, 1.0)).xyz;
  vC4 = (modelMatrix * vec4(c4, 1.0)).xyz;
  vC5 = (modelMatrix * vec4(c5, 1.0)).xyz;
  vSplitWay = splitWay;
  gl_Position = projectionMatrix * viewMatrix * vec4(vWorldPos, 1.0);
}
`;

const FRAGMENT_SHADER = /* glsl */`
precision highp float;

varying vec3 vWorldPos;
varying vec3 vC0;
varying vec3 vC1;
varying vec3 vC2;
varying vec3 vC3;
varying vec3 vC4;
varying vec3 vC5;
varying float vSplitWay;

uniform float uPatternScale;       // tiles per canonical unit
uniform float uPatternThickness;   // 0..1 fraction of cell occupied by content
uniform int uPattern;              // 0 = dots, 1 = bricks, 2 = checker
uniform float uStepSize;           // raymarch step size in world units
uniform int uMaxSteps;
uniform float uOpacity;
uniform vec3 uTintLow;
uniform vec3 uTintHigh;

// Geogram-convention orient_3d: positive iff (b-a, c-a, d-a) is right-handed.
float orient3d(vec3 a, vec3 b, vec3 c, vec3 d) {
  return determinant(mat3(b - a, c - a, d - a));
}

bool point_in_tet(vec3 p, vec3 T0, vec3 T1, vec3 T2, vec3 T3) {
  // Same four orient_3d checks as src/prism/predicates/inside_prism_tetra.cpp.
  // Tolerance widened slightly for shader-precision safety.
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

// Canonical unit prism coordinates of the 6 prism corners. Same as
// CANONICAL_PRISM in src/prism/phong/projection.cpp.
const vec3 CAN0 = vec3(0.0, 0.0, 0.0);
const vec3 CAN1 = vec3(1.0, 0.0, 0.0);
const vec3 CAN2 = vec3(0.0, 1.0, 0.0);
const vec3 CAN3 = vec3(0.0, 0.0, 1.0);
const vec3 CAN4 = vec3(1.0, 0.0, 1.0);
const vec3 CAN5 = vec3(0.0, 1.0, 1.0);

vec3 corner(int i) {
  return i == 0 ? vC0 :
         i == 1 ? vC1 :
         i == 2 ? vC2 :
         i == 3 ? vC3 :
         i == 4 ? vC4 : vC5;
}

vec3 canonicalCorner(int i) {
  return i == 0 ? CAN0 :
         i == 1 ? CAN1 :
         i == 2 ? CAN2 :
         i == 3 ? CAN3 :
         i == 4 ? CAN4 : CAN5;
}

// Decompose a world-space point inside the prism into canonical (u, v, t).
// Returns -1 if the point is not inside any of the prism's three tets.
//
// Mirrors prism::phong::phong_projection in src/prism/phong/projection.cpp.
bool decomposePrism(vec3 p, out vec3 uvt) {
  // TETRA_SPLIT_A = {{0,3,4,5}, {1,4,2,0}, {2,5,0,4}}
  // TETRA_SPLIT_B = {{0,3,4,5}, {1,4,5,0}, {2,5,0,1}}
  ivec4 tetA[3];
  tetA[0] = ivec4(0, 3, 4, 5);
  tetA[1] = ivec4(1, 4, 2, 0);
  tetA[2] = ivec4(2, 5, 0, 4);
  ivec4 tetB[3];
  tetB[0] = ivec4(0, 3, 4, 5);
  tetB[1] = ivec4(1, 4, 5, 0);
  tetB[2] = ivec4(2, 5, 0, 1);

  for (int i = 0; i < 3; ++i) {
    ivec4 t = vSplitWay > 0.5 ? tetA[i] : tetB[i];
    vec3 T0 = corner(t.x), T1 = corner(t.y), T2 = corner(t.z), T3 = corner(t.w);
    if (point_in_tet(p, T0, T1, T2, T3)) {
      vec4 b = baryTet(p, T0, T1, T2, T3);
      uvt = b.x * canonicalCorner(t.x) + b.y * canonicalCorner(t.y) +
            b.z * canonicalCorner(t.z) + b.w * canonicalCorner(t.w);
      return true;
    }
  }
  return false;
}

// 3D pattern in canonical coordinates. Returns rgba: rgb is content colour,
// a is its density at this point.
vec4 sampleShellTexture(vec3 uvt) {
  vec3 cell = fract(uvt * uPatternScale);
  vec3 d = abs(cell - 0.5);
  float r;

  if (uPattern == 0) {
    // Spheres on a regular grid.
    r = length(d);
    float density = smoothstep(uPatternThickness, uPatternThickness * 0.7, r);
    vec3 col = mix(uTintLow, uTintHigh, smoothstep(0.0, 0.6, length(uvt - 0.5)));
    return vec4(col, density);
  } else if (uPattern == 1) {
    // Stretcher-bond bricks: stagger every other v-row.
    float row = floor(uvt.y * uPatternScale);
    float u = uvt.x * uPatternScale + 0.5 * mod(row, 2.0);
    vec3 b = vec3(fract(u), fract(uvt.y * uPatternScale), fract(uvt.z * uPatternScale));
    vec3 ab = abs(b - 0.5);
    // Outer "mortar" gap.
    float mortar = step(0.5 - uPatternThickness * 0.05, max(ab.x, max(ab.y, ab.z)));
    vec3 col = mix(uTintLow, uTintHigh, b.z);
    return vec4(col, 1.0 - mortar);
  } else {
    // Tri-axial 3D checker.
    vec3 ic = floor(uvt * uPatternScale);
    float parity = mod(ic.x + ic.y + ic.z, 2.0);
    vec3 col = mix(uTintLow, uTintHigh, parity);
    // Soft cell boundaries via distance to nearest cell face.
    float wall = smoothstep(0.5 - uPatternThickness * 0.5,
                            0.5, max(d.x, max(d.y, d.z)));
    return vec4(col, mix(0.6, 0.05, wall));
  }
}

void main() {
  // Ray entry: front-facing fragment world position. Direction: away from camera.
  vec3 ro = vWorldPos;
  vec3 rd = normalize(vWorldPos - cameraPosition);

  vec4 acc = vec4(0.0);
  float t = 0.0;
  for (int i = 0; i < 256; ++i) {
    if (i >= uMaxSteps || acc.a > 0.97) break;
    vec3 p = ro + rd * t;
    vec3 uvt;
    if (!decomposePrism(p, uvt)) {
      // Stepped outside this prism's volume — done. (Rays that exit through
      // a side face into a neighbouring prism aren't continued; we'd need a
      // neighbour-link table for that, which adds complexity for limited
      // visual improvement at this pattern density.)
      break;
    }
    vec4 s = sampleShellTexture(uvt);
    s.a *= uOpacity;
    acc.rgb += s.rgb * s.a * (1.0 - acc.a);
    acc.a += s.a * (1.0 - acc.a);
    t += uStepSize;
  }

  if (acc.a < 0.01) discard;
  gl_FragColor = vec4(acc.rgb / max(acc.a, 0.001), acc.a);
}
`;

// Build (V, F, splitWays) per-prism arrays the shader expects, and wire them
// into an InstancedBufferGeometry + ShaderMaterial.
//
// midV / baseV / topV are Float64Array of length 3*N (PrismCage's per-vertex
// arrays after construction). F is Int32Array of length 3*M (face indices
// into the V arrays).
export function createShellMappingMesh(baseV, midV, topV, F, options = {}) {
  const opts = Object.assign({
    pattern: 0,            // 0 dots, 1 bricks, 2 checker
    patternScale: 8.0,
    patternThickness: 0.35,
    stepSize: 0.004,
    maxSteps: 64,
    opacity: 0.6,
    tintLow: new THREE.Color(0x4a8aff),
    tintHigh: new THREE.Color(0xff8a4a),
    useUpperSlab: true,    // mid->top by default; false picks base->mid
  }, options);

  const numPrisms = F.length / 3;

  // Per-vertex data for the unit prism boundary (shared across all instances).
  const vertsPerPrism = PRISM_TRIS.length * 3;
  const positions = new Float32Array(vertsPerPrism * 3);  // dummy; real pos is computed in VS
  const cornerIdx = new Float32Array(vertsPerPrism);
  for (let i = 0; i < PRISM_TRIS.length; i++) {
    for (let j = 0; j < 3; j++) {
      const idx = i * 3 + j;
      cornerIdx[idx] = PRISM_TRIS[i][j];
    }
  }

  // Per-instance corner positions. We pick which slab to render: upper
  // (mid -> top) or lower (base -> mid). The combined shell is two slabs,
  // each with its own bijective parametrisation.
  const c = [];  // c[k] is a Float32Array of length numPrisms*3 for corner k
  for (let k = 0; k < 6; k++) c.push(new Float32Array(numPrisms * 3));
  const splitWays = new Float32Array(numPrisms);

  const lowV = opts.useUpperSlab ? midV : baseV;
  const highV = opts.useUpperSlab ? topV : midV;

  for (let f = 0; f < numPrisms; f++) {
    const a = F[3 * f], b = F[3 * f + 1], cc = F[3 * f + 2];
    const slot = (k, v) => {
      c[k][3 * f] = v[0];
      c[k][3 * f + 1] = v[1];
      c[k][3 * f + 2] = v[2];
    };
    slot(0, [lowV[3*a], lowV[3*a+1], lowV[3*a+2]]);
    slot(1, [lowV[3*b], lowV[3*b+1], lowV[3*b+2]]);
    slot(2, [lowV[3*cc], lowV[3*cc+1], lowV[3*cc+2]]);
    slot(3, [highV[3*a], highV[3*a+1], highV[3*a+2]]);
    slot(4, [highV[3*b], highV[3*b+1], highV[3*b+2]]);
    slot(5, [highV[3*cc], highV[3*cc+1], highV[3*cc+2]]);
    // splitWay = (b > c) per src/prism/common.hpp::tetra_split_AorB.
    // After cage canonicalisation a is the minimum, so this matches.
    splitWays[f] = (b > cc) ? 1.0 : 0.0;
  }

  const geom = new THREE.InstancedBufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("cornerIdx", new THREE.BufferAttribute(cornerIdx, 1));
  geom.setAttribute("c0", new THREE.InstancedBufferAttribute(c[0], 3));
  geom.setAttribute("c1", new THREE.InstancedBufferAttribute(c[1], 3));
  geom.setAttribute("c2", new THREE.InstancedBufferAttribute(c[2], 3));
  geom.setAttribute("c3", new THREE.InstancedBufferAttribute(c[3], 3));
  geom.setAttribute("c4", new THREE.InstancedBufferAttribute(c[4], 3));
  geom.setAttribute("c5", new THREE.InstancedBufferAttribute(c[5], 3));
  geom.setAttribute("splitWay", new THREE.InstancedBufferAttribute(splitWays, 1));
  geom.instanceCount = numPrisms;

  // Conservative bbox so three.js doesn't try to frustum-cull instances.
  let minX=Infinity, minY=Infinity, minZ=Infinity;
  let maxX=-Infinity, maxY=-Infinity, maxZ=-Infinity;
  for (let i = 0; i < midV.length; i += 3) {
    minX = Math.min(minX, baseV[i], topV[i]);
    minY = Math.min(minY, baseV[i+1], topV[i+1]);
    minZ = Math.min(minZ, baseV[i+2], topV[i+2]);
    maxX = Math.max(maxX, baseV[i], topV[i]);
    maxY = Math.max(maxY, baseV[i+1], topV[i+1]);
    maxZ = Math.max(maxZ, baseV[i+2], topV[i+2]);
  }
  geom.boundingBox = new THREE.Box3(
    new THREE.Vector3(minX, minY, minZ),
    new THREE.Vector3(maxX, maxY, maxZ));
  geom.boundingSphere = geom.boundingBox.getBoundingSphere(new THREE.Sphere());

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uPatternScale: { value: opts.patternScale },
      uPatternThickness: { value: opts.patternThickness },
      uPattern: { value: opts.pattern },
      uStepSize: { value: opts.stepSize },
      uMaxSteps: { value: opts.maxSteps },
      uOpacity: { value: opts.opacity },
      uTintLow: { value: new THREE.Vector3(opts.tintLow.r, opts.tintLow.g, opts.tintLow.b) },
      uTintHigh: { value: new THREE.Vector3(opts.tintHigh.r, opts.tintHigh.g, opts.tintHigh.b) },
    },
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    side: THREE.FrontSide,
    transparent: true,
    depthWrite: false,
  });

  return new THREE.Mesh(geom, material);
}
