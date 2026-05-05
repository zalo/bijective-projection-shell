#include "intersections.hpp"

#include <igl/Hit.h>
extern "C" {
#include "igl/raytri.c"
}
#ifndef PRISM_NO_CGAL
#include "cgal/triangle_triangle_intersection.hpp"
#endif
#include <igl/barycentric_coordinates.h>

// Möller–Trumbore segment-triangle intersection. Returns the hit point if the
// segment crosses the triangle (interior or boundary), otherwise nullopt.
//
// This is a CGAL-free replacement for prism::cgal::segment_triangle_intersection;
// libigl's raytri.c implements the standard non-degenerate algorithm. Inputs
// are double-precision coordinates from a non-degenerate shell — exact-arith
// is not required for the call sites (smoothing fallback and feature-line
// intersection in `local_operations/smooth_pass.cpp` and the seg-vs-AABB
// path in `geogram/AABB.cpp`).
std::optional<Vec3d>
prism::intersections::segment_triangle_intersection_inexact(
    const std::array<Vec3d, 2> &seg, const std::array<Vec3d, 3> &tri) {
  double t = 0, u = 0, v = 0;
  auto s_d = seg[0];
  Vec3d dir = seg[1] - seg[0];
  int flag = intersect_triangle1(s_d.data(), dir.data(),
                                 const_cast<double *>(tri[0].data()),
                                 const_cast<double *>(tri[1].data()),
                                 const_cast<double *>(tri[2].data()),
                                 &t, &u, &v);
  if (flag != 1) return {};
  if (t < 0.0 || t > 1.0) return {};
  return s_d * (1.0 - t) + seg[1] * t;
}

bool prism::intersections::segment_triangle_hit(const std::array<Vec3d, 2> &seg,
                                                const std::array<Vec3d, 3> &tri,
                                                prism::Hit &hit) {
  hit.u = -1;
  hit.v = -1;
  hit.t = -1;
  auto [v0, v1, v2] = tri;
  auto s_d = seg[0];
  Vec3d dir = seg[1] - seg[0];
  auto flag = intersect_triangle1(s_d.data(), dir.data(), v0.data(), v1.data(),
                                  v2.data(), &hit.t, &hit.u, &hit.v);
  return (flag == 1 && hit.t >= 0);
}

bool prism::intersections::segment_triangle_hit_cgal(
    const std::array<Vec3d, 2> &seg, const std::array<Vec3d, 3> &tri,
    prism::Hit &hit) {
#ifdef PRISM_NO_CGAL
  auto inter = prism::intersections::segment_triangle_intersection_inexact(seg, tri);
#else
  auto inter = prism::cgal::segment_triangle_intersection(seg, tri);
#endif
  if (!inter) return false;
  Vec3d bc;
  igl::barycentric_coordinates(inter.value(), tri[0], tri[1], tri[2], bc);
  hit.u = bc[1];
  hit.v = bc[2];
  return true;
}