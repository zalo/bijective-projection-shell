#include "inside_octahedron.hpp"

#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>

#include "orient_robust.hpp"
#include "prism/predicates/triangle_triangle_intersection.hpp"
bool prism::inside_convex_octahedron(const std::array<Vec3d, 3>& base,
                                     const std::array<Vec3d, 3>& top,
                                     const Vec3d& point) {
  using prism::predicates::orient_3d;
  auto q = point.data();
  if (orient_3d(top[0].data(), top[1].data(), top[2].data(), q) >
          0 ||  // above top
      orient_3d(base[0].data(), base[1].data(), base[2].data(), q) <
          0)  // under base
    return false;
  for (short i = 0; i < 3; i++) {
    short i1 = (i + 1) % 3;
    auto b0 = base[i].data(), b1 = base[i1].data(), t0 = top[i].data(),
         t1 = top[i1].data();
    if (orient_3d(b0, t1, t0, b1) > 0) {  // b0-t1 out
      if (orient_3d(b0, t1, t0, q) > 0 || orient_3d(b0, b1, t1, q) > 0)
        return false;
    } else if (orient_3d(b0, b1, t0, q) > 0 || orient_3d(b1, t1, t0, q) > 0)
      return false;
  }
  return true;
}

void prism::determine_convex_octahedron(const std::array<Vec3d, 3>& base,
                                        const std::array<Vec3d, 3>& top,
                                        std::array<bool, 3>& oct_type,
                                        bool degenerate) {
  using prism::predicates::orient_3d;
  if (degenerate) {
    oct_type[0] = true;
    oct_type[2] = true;
    oct_type[1] = (orient_3d(base[1].data(), top[2].data(), top[1].data(),
                             base[2].data()) > 0);
    return;
  }
  for (short i0 = 0; i0 < 3; i0++) {
    short i1 = (i0 + 1) % 3;
    auto b0 = base[i0].data(), b1 = base[i1].data(), t0 = top[i0].data(),
         t1 = top[i1].data();
    oct_type[i0] = (orient_3d(b0, t1, t0, b1) > 0);
    // true: check b1-t0
  }
}
#include <spdlog/spdlog.h>

std::vector<Vec3i> oct_faces_from_type(const std::array<bool, 3>& oct_type,
                                       bool degenerate) {
  std::vector<Vec3i> oct_faces;
  oct_faces.reserve(8);
  oct_faces.push_back(Vec3i{0, 2, 1});  // bottom
  oct_faces.push_back(Vec3i{3, 4, 5});  // top
  for (short i = 0; i < 3; i++) {
    short i1 = (i + 1) % 3;
    if (oct_type[i]) {
      // 0 1 3, 1 4 3
      oct_faces.push_back(Vec3i{i, i1, i + 3});
      oct_faces.push_back(Vec3i{i1, i1 + 3, i + 3});
    } else {
      // 0 4 3, 0 1 4
      oct_faces.push_back(Vec3i{i, i1 + 3, i + 3});
      oct_faces.push_back(Vec3i{i, i1, i1 + 3});
    }
  }
  if (degenerate) {
    oct_faces.pop_back();
    oct_faces.erase(oct_faces.begin() + 2);
  }
  return std::move(oct_faces);
}

bool prism::triangle_intersect_octahedron(const std::array<Vec3d, 3>& base,
                                          const std::array<Vec3d, 3>& top,
                                          const std::array<bool, 3>& oct_type,
                                          const std::array<Vec3d, 3>& tri,
                                          bool degenerate) {
  auto oct_faces = oct_faces_from_type(oct_type, degenerate);
  std::array<Vec3d, 6> vecprism;
  std::array<Vec3d, 3> vectriangle;
  for (int i = 0; i < 3; i++) {
    vectriangle[i] = Vec3d(tri[i][0], tri[i][1], tri[i][2]);
    vecprism[i] = Vec3d(base[i][0], base[i][1], base[i][2]);
    vecprism[i + 3] = Vec3d(top[i][0], top[i][1], top[i][2]);
  }

  for (int i = 0; i < 3; i++) {  // for each point of tri
    bool point_inside = true;
    for (auto& j : oct_faces) {  // if outside any face, the point is out.
      if (prism::predicates::orient_3d(vecprism[j[0]].data(), vecprism[j[1]].data(),
                              vecprism[j[2]].data(),
                              vectriangle[i].data()) > 0) {  // i outside face j
        point_inside = false;
        break;
      }
    }
    if (point_inside) {
      return true;
    }  // any point inside or on
  }

  for (auto& j : oct_faces) {  // any face is intersecting
    std::array<Vec3d, 3> vecfacet{vecprism[j[0]], vecprism[j[1]],
                                  vecprism[j[2]]};
    if (prism::predicates::triangle_triangle_overlap(vectriangle, vecfacet)) {
      return true;
    }
  }

  return false;
}

bool prism::singularless_triangle_intersect_octahedron(
    const std::array<Vec3d, 3>& base, const std::array<Vec3d, 3>& top,
    const std::array<bool, 3>& oct_type, const std::array<Vec3d, 3>& tri) {
  // Triangle ABC vs. Pyramid AMN-APQ; vertex A is shared and ignored.
  // CGAL-free reimplementation using prism::predicates::*.
  auto oct_faces = oct_faces_from_type(oct_type, true);
  std::array<Vec3d, 6> vecprism;
  for (int i = 0; i < 3; i++) {
    vecprism[i] = base[i];
    vecprism[i + 3] = top[i];
  }
  const auto& vectriangle = tri;

  // Step 1.a: B or C inside the pyramid (strictly outside all faces -> not in)
  for (int i = 1; i < 3; i++) {
    bool point_inside = true;
    for (auto& j : oct_faces) {
      if (prism::predicates::orient_3d(
              vecprism[j[0]].data(), vecprism[j[1]].data(),
              vecprism[j[2]].data(), vectriangle[i].data()) > 0) {
        point_inside = false;
        break;
      }
    }
    if (point_inside) {
      spdlog::trace("Point In {}", i);
      return true;
    }
  }

  // Step 1.b: segment BC vs. each pyramid face.
  std::array<Vec3d, 2> BC{vectriangle[1], vectriangle[2]};
  for (auto& j : oct_faces) {
    std::array<Vec3d, 3> facet{vecprism[j[0]], vecprism[j[1]], vecprism[j[2]]};
    if (prism::predicates::segment_triangle_overlap(BC, facet)) {
      spdlog::trace("BC intersects pyramid face");
      return true;
    }
  }

  // Step 2: triangle ABC vs. pyramid base faces (oct_faces[3,4]).
  for (auto j : {3, 4}) {
    std::array<Vec3d, 3> base_facet{vecprism[oct_faces[j][0]],
                                    vecprism[oct_faces[j][1]],
                                    vecprism[oct_faces[j][2]]};
    if (prism::predicates::triangle_triangle_overlap(vectriangle, base_facet)) {
      spdlog::trace("base intersect ctri {}", j);
      return true;
    }
  }
  return false;
}

bool prism::pointless_triangle_intersect_octahedron(
    const std::array<Vec3d, 3>& base, const std::array<Vec3d, 3>& top,
    const std::array<bool, 3>& oct_type, const std::array<Vec3d, 3>& tri) {
  // will ignore 0 vs. 0
  auto ignore_id = -1;
  for (int i = 0; i < 3; i++) {
    if (base[i] == tri[0]) {
      ignore_id = i;
      break;
    }
    if (top[i] == tri[0]) {
      ignore_id = i + 3;
      break;
    }
  }
  assert(ignore_id >= 0);
  auto oct_faces = oct_faces_from_type(oct_type, false);
  std::array<Vec3d, 6> vecprism;
  auto& vectriangle = tri;
  for (int i = 0; i < 3; i++) {
    vecprism[i] = Vec3d(base[i][0], base[i][1], base[i][2]);
    vecprism[i + 3] = Vec3d(top[i][0], top[i][1], top[i][2]);
  }

  // Step 1. Test Segment BC vs. Octahedron
  // Step 1.a B or C inside octa
  for (int i = 1; i < 3; i++) {
    bool point_inside = true;
    for (auto& j : oct_faces) {  // if outside any face, the point is out.
      if (prism::predicates::orient_3d(vecprism[j[0]].data(), vecprism[j[1]].data(),
                              vecprism[j[2]].data(),
                              vectriangle[i].data()) > 0) {  // i outside face j
        point_inside = false;
        break;
      }
    }
    if (point_inside) {
      return true;
    }  // B/C point inside or on
  }
  // Step 1.b Segment BC intersect octa-faces adjacent to A
  // and Triangle ABC intersect octa-faces non-adjacent to A
  for (auto& j : oct_faces) {  // any face is intersecting
    std::array<Vec3d, 3> vecfacet{vecprism[j[0]], vecprism[j[1]],
                                  vecprism[j[2]]};
    if (j[0] == ignore_id || j[1] == ignore_id || j[2] == ignore_id) {
      if (prism::predicates::segment_triangle_overlap(
              {vectriangle[1], vectriangle[2]}, vecfacet)) {
        return true;
      }
    } else {
      if (prism::predicates::triangle_triangle_overlap(vectriangle, vecfacet)) {
        return true;
      }
    }
  }

  // Otherwise, not intersecting.
  return false;
}