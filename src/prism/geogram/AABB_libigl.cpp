// libigl-backed implementation of prism::geogram::AABB. Used when
// PRISM_NO_CGAL is defined (also implies no geogram). Same public API as
// AABB.cpp; replaces the geogram MeshFacetsAABB tree + CGAL exact triangle/
// segment intersections with libigl's templated AABB and Möller–Trumbore.

#include "AABB.hpp"

#include <igl/AABB.h>
#include <igl/Hit.h>
#include <igl/barycentric_coordinates.h>
#include <igl/per_face_normals.h>
#include <spdlog/spdlog.h>

#include <limits>
#include <prism/intersections.hpp>
#include <prism/predicates/triangle_triangle_intersection.hpp>

namespace prism::geogram {

// We stash the libigl tree, the geometry, and the box-overlap helper inside
// a private struct hidden behind GEO::MeshFacetsAABB-shaped shared_ptrs in
// the header — repurpose those slots so the existing header stays binary
// compatible enough not to require touching call sites.
//
// `geo_tree_ptr_` holds an igl::AABB tree (typed `void*`-style via shared_ptr<void>).
// `geo_polyhedron_ptr_` is unused.
struct LibIglAABB {
  igl::AABB<RowMatd, 3> tree;
  RowMatd V;
  RowMati F;
};

namespace {
// Use a shared registry keyed by AABB instance to attach the LibIglAABB
// (since the header's geo_tree_ptr_ has a fixed pointer type and we don't
// want to change the header). Map raw pointer → impl.
std::map<const AABB*, std::shared_ptr<LibIglAABB>>& registry() {
  static std::map<const AABB*, std::shared_ptr<LibIglAABB>> r;
  return r;
}
const LibIglAABB& impl_of(const AABB* self) {
  auto it = registry().find(self);
  return *it->second;
}
}  // namespace

AABB::AABB(const RowMatd& V, const RowMati& F, bool _enabled) : enabled(_enabled) {
  if (!enabled) return;
  auto inst = std::make_shared<LibIglAABB>();
  inst->V = V;
  inst->F = F;
  inst->tree.init(inst->V, inst->F);
  registry()[this] = inst;

  geo_vertex_ind.resize(V.rows());
  for (int i = 0; i < V.rows(); i++) geo_vertex_ind[i] = i;
  geo_face_ind.resize(F.rows());
  for (int i = 0; i < F.rows(); i++) geo_face_ind[i] = i;
}

bool AABB::intersects_triangle(const std::array<Vec3d, 3>& P,
                               bool use_freeze) const {
  if (!enabled) return false;
  const auto& self = impl_of(this);

  // Find candidate triangles whose AABBs overlap the input triangle's AABB.
  Eigen::AlignedBox<double, 3> qbox;
  for (int i = 0; i < 3; i++) {
    qbox.extend(P[i].transpose());
  }

  std::vector<int> cand;
  // The libigl AABB doesn't expose a direct box-overlap query, but
  // squared_distance with a sentinel and `intersect` flow gets ugly. Use a
  // manual recursive walk instead — fast and small.
  std::function<void(const igl::AABB<RowMatd, 3>*)> walk =
      [&](const igl::AABB<RowMatd, 3>* node) {
        if (!node) return;
        const auto& bb = node->m_box;
        if (!bb.intersects(qbox)) return;
        if (node->is_leaf()) {
          cand.push_back(node->m_primitive);
          return;
        }
        walk(node->m_left);
        walk(node->m_right);
      };
  walk(&self.tree);

  for (int f : cand) {
    Vec3d v0 = self.V.row(self.F(f, 0));
    Vec3d v1 = self.V.row(self.F(f, 1));
    Vec3d v2 = self.V.row(self.F(f, 2));
    std::array<Vec3d, 3> kt{v0, v1, v2};
    if (use_freeze && self.F(f, 0) < num_freeze) {
      Vec3d kp0 = P[0], kp1 = P[1], kp2 = P[2];
      Vec3d kv0 = v0, kv1 = v1, kv2 = v2;
      if (kv1 == kp0) std::swap(kv1, kv0);
      if (kv2 == kp0) std::swap(kv2, kv0);
      std::array<Vec3d, 2> ks{kv1, kv2};
      std::array<Vec3d, 3> ktt{kv0, kv1, kv2};
      std::array<Vec3d, 2> ks1{kp1, kp2};
      if (prism::predicates::segment_triangle_overlap(ks, P) ||
          prism::predicates::segment_triangle_overlap(ks1, ktt)) {
        return true;
      }
      continue;
    }
    if (prism::predicates::triangle_triangle_overlap(kt, P)) return true;
  }
  return false;
}

bool AABB::segment_query(const Vec3d& start, const Vec3d& end, int& face_id,
                         Vec3d& finalpoint) const {
  if (!enabled) {
    face_id = -1;
    return false;
  }
  const auto& self = impl_of(this);

  // Box-prefilter: walk the tree to find faces whose AABBs overlap the
  // segment's AABB, then test each candidate with Möller–Trumbore.
  Eigen::AlignedBox<double, 3> qbox;
  qbox.extend(start.transpose());
  qbox.extend(end.transpose());
  std::vector<int> cand;
  std::function<void(const igl::AABB<RowMatd, 3>*)> walk =
      [&](const igl::AABB<RowMatd, 3>* node) {
        if (!node) return;
        if (!node->m_box.intersects(qbox)) return;
        if (node->is_leaf()) {
          cand.push_back(node->m_primitive);
          return;
        }
        walk(node->m_left);
        walk(node->m_right);
      };
  walk(&self.tree);

  std::array<Vec3d, 2> seg{start, end};
  face_id = -1;
  for (int f : cand) {
    Vec3d v0 = self.V.row(self.F(f, 0));
    Vec3d v1 = self.V.row(self.F(f, 1));
    Vec3d v2 = self.V.row(self.F(f, 2));
    auto inter =
        prism::intersections::segment_triangle_intersection_inexact(seg, {v0, v1, v2});
    if (inter) {
      finalpoint = *inter;
      face_id = f;
      return true;
    }
  }
  return false;
}

std::optional<Vec3d> AABB::segment_query(const Vec3d& start,
                                         const Vec3d& end) const {
  int face_id = -1;
  Vec3d finalpoint;
  if (segment_query(start, end, face_id, finalpoint)) return finalpoint;
  return {};
}

bool AABB::segment_hit(const Vec3d& start, const Vec3d& end,
                       prism::Hit& hit) const {
  int fid = -1;
  Vec3d finalpoint;
  if (!segment_query(start, end, fid, finalpoint)) return false;
  const auto& self = impl_of(this);
  Vec3d v0 = self.V.row(self.F(fid, 0));
  Vec3d v1 = self.V.row(self.F(fid, 1));
  Vec3d v2 = self.V.row(self.F(fid, 2));
  // Triangle areas via cross products.
  auto area = [](const Vec3d& a, const Vec3d& b, const Vec3d& c) {
    return 0.5 * (b - a).cross(c - a).norm();
  };
  double a0 = area(finalpoint, v1, v2);
  double a1 = area(finalpoint, v2, v0);
  double a2 = area(finalpoint, v0, v1);
  double tot = a0 + a1 + a2;
  hit.u = a1 / tot;
  hit.v = a2 / tot;
  hit.id = geo_face_ind[fid];
  hit.gid = 0;
  return true;
}

double AABB::ray_length(const Vec3d& start, const Vec3d& dir, double max_step,
                        int ignore_v) const {
  if (!enabled) return max_step;
  const auto& self = impl_of(this);
  Vec3d end = start + max_step * dir;
  Eigen::AlignedBox<double, 3> qbox;
  qbox.extend(start.transpose());
  qbox.extend(end.transpose());
  std::vector<int> cand;
  std::function<void(const igl::AABB<RowMatd, 3>*)> walk =
      [&](const igl::AABB<RowMatd, 3>* node) {
        if (!node) return;
        if (!node->m_box.intersects(qbox)) return;
        if (node->is_leaf()) {
          cand.push_back(node->m_primitive);
          return;
        }
        walk(node->m_left);
        walk(node->m_right);
      };
  walk(&self.tree);

  std::array<Vec3d, 2> seg{start, end};
  double min_dist_sq = max_step * max_step;
  for (int f : cand) {
    int i0 = self.F(f, 0), i1 = self.F(f, 1), i2 = self.F(f, 2);
    if (i0 == ignore_v || i1 == ignore_v || i2 == ignore_v) continue;
    Vec3d v0 = self.V.row(i0);
    Vec3d v1 = self.V.row(i1);
    Vec3d v2 = self.V.row(i2);
    auto inter =
        prism::intersections::segment_triangle_intersection_inexact(seg, {v0, v1, v2});
    if (inter) {
      double d_sq = (*inter - start).squaredNorm();
      min_dist_sq = std::min(min_dist_sq, d_sq);
    }
  }
  return std::sqrt(min_dist_sq);
}

bool AABB::numerical_self_intersection(double tol) const {
  if (!enabled) return false;
  const auto& self = impl_of(this);
  // For each vertex, find triangles whose AABB is within tol; if any such
  // triangle is not adjacent, report a near-self-intersection.
  for (int v = 0; v < self.V.rows(); v++) {
    Eigen::AlignedBox<double, 3> qbox;
    Vec3d p = self.V.row(v);
    qbox.extend((p.transpose().array() - tol).matrix());
    qbox.extend((p.transpose().array() + tol).matrix());

    std::vector<int> cand;
    std::function<void(const igl::AABB<RowMatd, 3>*)> walk =
        [&](const igl::AABB<RowMatd, 3>* node) {
          if (!node) return;
          if (!node->m_box.intersects(qbox)) return;
          if (node->is_leaf()) {
            cand.push_back(node->m_primitive);
            return;
          }
          walk(node->m_left);
          walk(node->m_right);
        };
    walk(&self.tree);

    for (int f : cand) {
      int i0 = self.F(f, 0), i1 = self.F(f, 1), i2 = self.F(f, 2);
      if (i0 == v || i1 == v || i2 == v) continue;
      // distance point→triangle: project p onto triangle, clamp to bary.
      Vec3d a = self.V.row(i0), b = self.V.row(i1), c = self.V.row(i2);
      Vec3d ab = b - a, ac = c - a, ap = p - a;
      double d1 = ab.dot(ap), d2 = ac.dot(ap);
      double d3 = ab.dot(p - b), d4 = ac.dot(p - b);
      double d5 = ab.dot(p - c), d6 = ac.dot(p - c);
      Vec3d closest;
      if (d1 <= 0 && d2 <= 0) closest = a;
      else if (d3 >= 0 && d4 <= d3) closest = b;
      else if (d6 >= 0 && d5 <= d6) closest = c;
      else {
        double vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) {
          double tparam = d1 / (d1 - d3);
          closest = a + tparam * ab;
        } else {
          double vb = d5 * d2 - d1 * d6;
          if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            double tparam = d2 / (d2 - d6);
            closest = a + tparam * ac;
          } else {
            double va = d3 * d6 - d5 * d4;
            if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
              double tparam = (d4 - d3) / ((d4 - d3) + (d5 - d6));
              closest = b + tparam * (c - b);
            } else {
              double denom = 1.0 / (va + vb + vc);
              double bv = vb * denom, bw = vc * denom;
              closest = a + ab * bv + ac * bw;
            }
          }
        }
      }
      if ((p - closest).squaredNorm() <= tol * tol) return true;
    }
  }
  return false;
}

}  // namespace prism::geogram
