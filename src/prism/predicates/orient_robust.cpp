#include "orient_robust.hpp"

#include <mutex>

#include <robust_predicates/predicates.h>

namespace {
std::once_flag g_init_flag;
inline void init_once() { std::call_once(g_init_flag, [] { ::exactinit(); }); }
inline int sgn(double v) { return (v > 0) - (v < 0); }
}  // namespace

namespace prism {
namespace predicates {

int orient_3d(const double* p0, const double* p1, const double* p2,
              const double* p3) {
  init_once();
  // Shewchuk returns sign(det([p0-p3; p1-p3; p2-p3])), which is the negative
  // of geogram's sign(det([p1-p0; p2-p0; p3-p0])). Negate.
  double r = ::orient3d(const_cast<double*>(p0), const_cast<double*>(p1),
                        const_cast<double*>(p2), const_cast<double*>(p3));
  return -sgn(r);
}

int orient_2d(const double* p0, const double* p1, const double* p2) {
  init_once();
  // Shewchuk's orient2d already matches geogram's convention.
  double r = ::orient2d(const_cast<double*>(p0), const_cast<double*>(p1),
                        const_cast<double*>(p2));
  return sgn(r);
}

}  // namespace predicates
}  // namespace prism
