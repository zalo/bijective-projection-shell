// Shell-mapping heightmap raymarcher for the bijective prismatic shell.
//
// Renders the upper slab of each PrismCage face (mid → top) as the prism's
// outer boundary. In the fragment shader, every fragment's world position is
// the ray's entry into the prism. We raymarch *inward* along the view
// direction, decompose each sample into canonical-prism coordinates
// (u, v, t) via the same Phong projection as src/prism/phong/projection.cpp,
// and at each step compare t against a procedural heightmap h(u, v).
//
// When t crosses below h(u, v), the ray has hit the displaced surface;
// shade with a cheap lambert-from-central-differences normal and return.
//
// This is the classical Porumbescu et al. 2005 "shell maps" rendering
// adapted to PrismCage's prismatic parametrisation. Because the (u, v, t)
// bijection is exact, the heightmap follows surface curvature without
// distortion.

import * as THREE from "three";

// We render ONLY the top triangle (vertices 3,4,5 = top[a], top[b], top[c]).
// Side rectangles are intentionally omitted: at silhouettes they'd render the
// prism's side wall, where the heightmap raymarch enters tangentially and
// often falls back to the valley shade — visible as black walls between
// neighbouring prisms. With only the top, every fragment enters through the
// outer surface (canonical t = 1) and marches inward to t = 0; for a closed
// input mesh that's all the camera ever sees from outside.
const PRISM_TRIS = [
  [3, 4, 5],
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

uniform float uPatternScale;       // tiles per canonical (u, v) unit
uniform float uBumpHeight;         // max h(u,v) value, 0..1
uniform int uPattern;              // 0 hex, 1 bricks, 2 fbm
uniform float uStepSize;           // raymarch step in world units
uniform int uMaxSteps;
uniform vec3 uTintLow;
uniform vec3 uTintHigh;
uniform vec3 uLightDir;            // world-space directional light

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

bool decomposePrism(vec3 p, out vec3 uvt) {
  // TETRA_SPLIT_A = {{0,3,4,5}, {1,4,2,0}, {2,5,0,4}}
  // TETRA_SPLIT_B = {{0,3,4,5}, {1,4,5,0}, {2,5,0,1}}
  ivec4 tetA0 = ivec4(0, 3, 4, 5);
  ivec4 tetA1 = ivec4(1, 4, 2, 0);
  ivec4 tetA2 = ivec4(2, 5, 0, 4);
  ivec4 tetB0 = ivec4(0, 3, 4, 5);
  ivec4 tetB1 = ivec4(1, 4, 5, 0);
  ivec4 tetB2 = ivec4(2, 5, 0, 1);

  for (int i = 0; i < 3; ++i) {
    ivec4 t;
    if (vSplitWay > 0.5) {
      t = i == 0 ? tetA0 : (i == 1 ? tetA1 : tetA2);
    } else {
      t = i == 0 ? tetB0 : (i == 1 ? tetB1 : tetB2);
    }
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

// ---- 2D procedural heightfield h(u, v) ∈ [0, 1] ----

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
    p *= 2.0;
    a *= 0.5;
  }
  return v;
}

// Hex bumps in 2D — used inside triplanar.
float hexBumps2D(vec2 uv) {
  vec2 s = uv * vec2(uPatternScale * 1.732, uPatternScale);
  vec2 h1 = vec2(s.x, s.y);
  vec2 h2 = vec2(s.x + uPatternScale * 0.866, s.y + 0.5);
  vec2 c1 = h1 - (floor(h1) + 0.5);
  vec2 c2 = h2 - (floor(h2) + 0.5);
  float d = min(length(c1), length(c2));
  return smoothstep(0.5, 0.15, d);
}

// Stretcher-bond bricks in 2D.
float brickBumps2D(vec2 uv) {
  float row = floor(uv.y * uPatternScale);
  vec2 b = vec2(uv.x * uPatternScale + 0.5 * mod(row, 2.0),
                uv.y * uPatternScale);
  vec2 c = abs(fract(b) - 0.5);
  float gap = 0.5 - max(c.x, c.y);
  return smoothstep(0.0, 0.08, gap);
}

float pattern2D(vec2 uv) {
  if (uPattern == 0) return hexBumps2D(uv);
  if (uPattern == 1) return brickBumps2D(uv);
  return fbm(uv * uPatternScale);
}

// Triplanar evaluation: sample the 2D pattern in three orthogonal planes
// and blend by squared world-space normal weights. Drives all three
// patterns from a single 3D world-space coordinate, so they tile
// continuously across adjacent prisms even though each prism's local
// (u, v) parameterisation is discontinuous at its triangular boundary.
float triplanar(vec3 p, vec3 n) {
  vec3 w = abs(normalize(n));
  w = pow(w, vec3(4.0));         // sharper blend
  w /= (w.x + w.y + w.z + 1e-6);
  return w.x * pattern2D(p.yz)
       + w.y * pattern2D(p.zx)
       + w.z * pattern2D(p.xy);
}

float heightmap(vec3 worldP, vec3 nWorld) {
  return triplanar(worldP, nWorld) * uBumpHeight;
}

// The bijective image of (u, v) on the mid surface. For the upper slab,
// vC0..vC2 are mid corners (canonical t = 0).
vec3 midImage(vec3 uvt) {
  return (1.0 - uvt.x - uvt.y) * vC0 + uvt.x * vC1 + uvt.y * vC2;
}

void main() {
  vec3 ro = vWorldPos;
  vec3 rd = normalize(vWorldPos - cameraPosition);

  // Surface normal estimate — the prism's pillar (mid-to-top vector). Used
  // as the triplanar blending normal so every fragment in this prism uses
  // a consistent projection axis.
  vec3 nSurface = normalize(vC3 - vC0);

  bool everInside = false;
  vec3 lastUvt = vec3(0.5, 0.5, 0.0);
  for (int i = 0; i < 256; ++i) {
    if (i >= uMaxSteps) break;
    vec3 p = ro + rd * (float(i) * uStepSize);
    vec3 uvt;
    if (!decomposePrism(p, uvt)) break;
    everInside = true;
    lastUvt = uvt;

    // Heightmap is driven by the bijective IMAGE on the mid surface — a
    // 3D world-space coordinate that's continuous across prism boundaries
    // (because the original mesh shares vertices/edges between faces).
    // This kills the "different stretching per prism" artefact: the same
    // world-space point gives the same height regardless of which prism
    // the ray is currently traversing.
    vec3 imgP = midImage(uvt);
    float h = heightmap(imgP, nSurface);
    if (uvt.z <= h) {
      const float dx = 0.005;
      float hu = heightmap(midImage(uvt + vec3(dx, 0, 0)), nSurface);
      float hv = heightmap(midImage(uvt + vec3(0, dx, 0)), nSurface);
      vec3 n_canon = normalize(vec3(-(hu - h) / dx, -(hv - h) / dx, 1.0));
      vec3 du = normalize(vC1 - vC0);
      vec3 dv = normalize(vC2 - vC0);
      vec3 n_world = normalize(n_canon.x * du + n_canon.y * dv + n_canon.z * nSurface);

      float diffuse = max(0.15, dot(n_world, normalize(uLightDir)));
      vec3 col = mix(uTintLow, uTintHigh,
                     smoothstep(0.0, 1.0, h / max(uBumpHeight, 1e-6)));
      gl_FragColor = vec4(col * diffuse, 1.0);
      return;
    }
  }

  // Ray exited the prism without hitting. Shade as the "valley floor" —
  // smooth surface lit by the directional light. Uses the brightest end of
  // the colour ramp to avoid the dark-walls artefact at silhouettes.
  if (!everInside) discard;
  float diffuse = max(0.15, dot(nSurface, normalize(uLightDir)));
  gl_FragColor = vec4(mix(uTintLow, uTintHigh, 0.4) * diffuse, 1.0);
}
`;

export function createShellMappingMesh(baseV, midV, topV, F, options = {}) {
  const opts = Object.assign({
    pattern: 0,            // 0 hex, 1 bricks, 2 fbm
    patternScale: 8.0,
    bumpHeight: 0.7,
    stepsPerSlab: 64,      // samples to traverse one prism's thickness
    maxSteps: 192,
    tintLow: new THREE.Color(0x2a4060),
    tintHigh: new THREE.Color(0xffd28a),
    lightDir: new THREE.Vector3(0.5, 0.7, 0.5),
  }, options);

  const numPrisms = F.length / 3;

  const vertsPerPrism = PRISM_TRIS.length * 3;
  const positions = new Float32Array(vertsPerPrism * 3);
  const cornerIdx = new Float32Array(vertsPerPrism);
  for (let i = 0; i < PRISM_TRIS.length; i++) {
    for (let j = 0; j < 3; j++) {
      cornerIdx[i * 3 + j] = PRISM_TRIS[i][j];
    }
  }

  const c = [];
  for (let k = 0; k < 6; k++) c.push(new Float32Array(numPrisms * 3));
  const splitWays = new Float32Array(numPrisms);

  // Heightmap shell mapping renders the *upper* slab only (mid -> top), with
  // bumps protruding from the surface (mid) outward. The lower slab is
  // omitted; if you wanted two-sided displacement it would need its own
  // pass with reversed t.
  for (let f = 0; f < numPrisms; f++) {
    const a = F[3 * f], b = F[3 * f + 1], cc = F[3 * f + 2];
    const slot = (k, v) => {
      c[k][3 * f] = v[0];
      c[k][3 * f + 1] = v[1];
      c[k][3 * f + 2] = v[2];
    };
    // mid -> corners 0..2 (canonical t = 0)
    slot(0, [midV[3*a], midV[3*a+1], midV[3*a+2]]);
    slot(1, [midV[3*b], midV[3*b+1], midV[3*b+2]]);
    slot(2, [midV[3*cc], midV[3*cc+1], midV[3*cc+2]]);
    // top -> corners 3..5 (canonical t = 1)
    slot(3, [topV[3*a], topV[3*a+1], topV[3*a+2]]);
    slot(4, [topV[3*b], topV[3*b+1], topV[3*b+2]]);
    slot(5, [topV[3*cc], topV[3*cc+1], topV[3*cc+2]]);
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

  let minX=Infinity, minY=Infinity, minZ=Infinity;
  let maxX=-Infinity, maxY=-Infinity, maxZ=-Infinity;
  // Median per-pillar thickness (mid -> top), used to set raymarch step.
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
  // Step size = slab_thickness / stepsPerSlab. Cosine-of-grazing-angle
  // factor of 1/2 is folded into maxSteps so an oblique ray still finishes.
  const stepSize = Math.max(medianPillar / opts.stepsPerSlab, 1e-4);
  geom.boundingBox = new THREE.Box3(
    new THREE.Vector3(minX, minY, minZ),
    new THREE.Vector3(maxX, maxY, maxZ));
  geom.boundingSphere = geom.boundingBox.getBoundingSphere(new THREE.Sphere());

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uPatternScale: { value: opts.patternScale },
      uBumpHeight: { value: opts.bumpHeight },
      uPattern: { value: opts.pattern },
      uStepSize: { value: stepSize },
      uMaxSteps: { value: opts.maxSteps },
      uTintLow: { value: new THREE.Vector3(opts.tintLow.r, opts.tintLow.g, opts.tintLow.b) },
      uTintHigh: { value: new THREE.Vector3(opts.tintHigh.r, opts.tintHigh.g, opts.tintHigh.b) },
      uLightDir: { value: opts.lightDir.clone() },
    },
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    side: THREE.FrontSide,
  });

  return new THREE.Mesh(geom, material);
}
