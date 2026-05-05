// WASM binding for the full prism_library (PRISM_WASM=ON build).
//
// Builds a real PrismCage via the in-source SIGGRAPH Asia 2020 pipeline (vertex
// normal selection + offset cage construction), exposes its base/mid/top
// vertices and faces to JS, and provides the bijective Phong projection on
// arbitrary query points lying inside the prismatic shell.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include <prism/PrismCage.hpp>
#include <prism/common.hpp>
#include <prism/geogram/AABB.hpp>
#include <prism/phong/projection.hpp>
#include <prism/predicates/inside_prism_tetra.hpp>
#include <prism/spatial-hash/AABB_hash.hpp>

namespace {
using prism::predicates::point_in_tetrahedron;

constexpr std::array<std::array<size_t, 4>, 3> TETRA_SPLIT_A_LOCAL{
    {{0, 3, 4, 5}, {1, 4, 2, 0}, {2, 5, 0, 4}}};
constexpr std::array<std::array<size_t, 4>, 3> TETRA_SPLIT_B_LOCAL{
    {{0, 3, 4, 5}, {1, 4, 5, 0}, {2, 5, 0, 1}}};

bool point_in_prism(const std::array<Vec3d, 6>& V, bool tetra_split_way,
                    const Vec3d& point) {
  const auto& tet = tetra_split_way ? TETRA_SPLIT_A_LOCAL : TETRA_SPLIT_B_LOCAL;
  for (int i = 0; i < 3; ++i) {
    if (point_in_tetrahedron(point, V[tet[i][0]], V[tet[i][1]], V[tet[i][2]],
                             V[tet[i][3]]))
      return true;
  }
  return false;
}
}  // anonymous

namespace prism_wasm {

struct ShellResult {
  std::vector<double> midV;   // n*3 (original surface, after pre-processing)
  std::vector<double> baseV;  // n*3
  std::vector<double> topV;   // n*3
  std::vector<int> F;          // m*3
  int numFreeze = 0;
};

struct ProjectionResult {
  std::vector<int> faceId;     // n queries
  std::vector<double> uvt;     // n*3 (u,v on the shell triangle, t = height)
  std::vector<double> imageP;  // n*3 -> projected point on the original surface
  std::vector<int> hit;        // 0/1
  std::vector<int> stratum;    // 0 = lower slab (base..mid), 1 = upper slab (mid..top)
};

static std::shared_ptr<PrismCage> g_cage;

// Build a PrismCage from (V, F). Returns the resulting base/mid/top/F arrays.
// `thicknessRatio` -> doosabineps (target thickness as fraction of bbox).
// `initialStep` -> initial cage thickness.
ShellResult buildShell(emscripten::val Vjs, emscripten::val Fjs,
                       double thicknessRatio, double initialStep) {
  std::vector<double> Vd =
      emscripten::convertJSArrayToNumberVector<double>(Vjs);
  std::vector<int> Fi = emscripten::convertJSArrayToNumberVector<int>(Fjs);
  size_t nV = Vd.size() / 3;
  size_t nF = Fi.size() / 3;

  RowMatd V(nV, 3);
  RowMati F(nF, 3);
  for (size_t i = 0; i < nV; ++i) V.row(i) << Vd[3 * i], Vd[3 * i + 1], Vd[3 * i + 2];
  for (size_t i = 0; i < nF; ++i) F.row(i) << Fi[3 * i], Fi[3 * i + 1], Fi[3 * i + 2];

  g_cage = std::make_shared<PrismCage>(V, F, thicknessRatio, initialStep,
                                       PrismCage::SeparateType::kSurface);

  ShellResult r;
  size_t nVOut = g_cage->mid.size();
  r.midV.resize(nVOut * 3);
  r.baseV.resize(nVOut * 3);
  r.topV.resize(nVOut * 3);
  for (size_t i = 0; i < nVOut; ++i) {
    for (int k = 0; k < 3; ++k) {
      r.midV[3 * i + k] = g_cage->mid[i][k];
      r.baseV[3 * i + k] = g_cage->base[i][k];
      r.topV[3 * i + k] = g_cage->top[i][k];
    }
  }
  r.F.resize(g_cage->F.size() * 3);
  for (size_t i = 0; i < g_cage->F.size(); ++i) {
    r.F[3 * i + 0] = g_cage->F[i][0];
    r.F[3 * i + 1] = g_cage->F[i][1];
    r.F[3 * i + 2] = g_cage->F[i][2];
  }
  r.numFreeze = g_cage->ref.aabb ? g_cage->ref.aabb->num_freeze : 0;
  return r;
}

// Project a list of query points through the current shell.
//
// For each query point q:
//   1. Linear scan over each face's prism (lower slab base→mid and upper slab
//      mid→top). The prism that contains q is found via point_in_prism.
//   2. phong_projection returns canonical (u, v, t) inside the slab.
//   3. The image on the original surface is recovered from the mid triangle
//      with bary coords (1-u-v, u, v).
//
// The lower slab maps to height fraction t in [0, 0.5] and the upper slab to
// [0.5, 1.0] for visualisation.
ProjectionResult projectPoints(emscripten::val Qjs) {
  std::vector<double> Q = emscripten::convertJSArrayToNumberVector<double>(Qjs);
  size_t nQ = Q.size() / 3;
  ProjectionResult r;
  r.faceId.assign(nQ, -1);
  r.uvt.assign(nQ * 3, 0.0);
  r.imageP.assign(nQ * 3, 0.0);
  r.hit.assign(nQ, 0);
  r.stratum.assign(nQ, -1);
  if (!g_cage) return r;

  size_t nF = g_cage->F.size();

  for (size_t q = 0; q < nQ; ++q) {
    Vec3d p(Q[3 * q], Q[3 * q + 1], Q[3 * q + 2]);
    bool found = false;
    for (size_t f = 0; f < nF && !found; ++f) {
      auto& abc = g_cage->F[f];
      int v0 = abc[0], v1 = abc[1], v2 = abc[2];
      bool tetra_split_way = v1 > v2;  // tetra_split_AorB convention

      // Lower slab: base → mid
      std::array<Vec3d, 6> lower{g_cage->base[v0], g_cage->base[v1],
                                 g_cage->base[v2], g_cage->mid[v0],
                                 g_cage->mid[v1],  g_cage->mid[v2]};
      // Upper slab: mid → top
      std::array<Vec3d, 6> upper{g_cage->mid[v0], g_cage->mid[v1],
                                 g_cage->mid[v2], g_cage->top[v0],
                                 g_cage->top[v1], g_cage->top[v2]};

      std::array<double, 3> tuple{};
      bool hit_lower = false, hit_upper = false;
      if (point_in_prism(lower, tetra_split_way, p)) {
        if (prism::phong::phong_projection(lower, p, tetra_split_way, tuple))
          hit_lower = true;
      }
      if (!hit_lower && point_in_prism(upper, tetra_split_way, p)) {
        if (prism::phong::phong_projection(upper, p, tetra_split_way, tuple))
          hit_upper = true;
      }

      if (hit_lower || hit_upper) {
        double u = tuple[0], v = tuple[1], t_local = tuple[2];
        Vec3d ma = g_cage->mid[v0], mb = g_cage->mid[v1], mc = g_cage->mid[v2];
        Vec3d image = ma * (1.0 - u - v) + mb * u + mc * v;
        r.faceId[q] = (int)f;
        r.uvt[3 * q] = u;
        r.uvt[3 * q + 1] = v;
        r.uvt[3 * q + 2] = hit_lower ? 0.5 * t_local : 0.5 + 0.5 * t_local;
        r.imageP[3 * q] = image[0];
        r.imageP[3 * q + 1] = image[1];
        r.imageP[3 * q + 2] = image[2];
        r.hit[q] = 1;
        r.stratum[q] = hit_lower ? 0 : 1;
        found = true;
      }
    }
  }
  return r;
}

}  // namespace prism_wasm

EMSCRIPTEN_BINDINGS(prism_full_wasm) {
  using namespace emscripten;
  value_object<prism_wasm::ShellResult>("ShellResult")
      .field("midV", &prism_wasm::ShellResult::midV)
      .field("baseV", &prism_wasm::ShellResult::baseV)
      .field("topV", &prism_wasm::ShellResult::topV)
      .field("F", &prism_wasm::ShellResult::F)
      .field("numFreeze", &prism_wasm::ShellResult::numFreeze);

  value_object<prism_wasm::ProjectionResult>("ProjectionResult")
      .field("faceId", &prism_wasm::ProjectionResult::faceId)
      .field("uvt", &prism_wasm::ProjectionResult::uvt)
      .field("imageP", &prism_wasm::ProjectionResult::imageP)
      .field("hit", &prism_wasm::ProjectionResult::hit)
      .field("stratum", &prism_wasm::ProjectionResult::stratum);

  register_vector<double>("VectorDouble");
  register_vector<int>("VectorInt");

  function("buildShell", &prism_wasm::buildShell);
  function("projectPoints", &prism_wasm::projectPoints);
}
