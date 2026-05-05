#ifndef PRISM_PREDICATES_ORIENT_ROBUST_HPP
#define PRISM_PREDICATES_ORIENT_ROBUST_HPP

// Drop-in replacement for geogram's GEO::PCK::orient_2d / orient_3d.
//
// Backed by Shewchuk's adaptive-precision predicates (external/robust_predicates).
// Geogram returns sign(det([p1-p0; p2-p0; p3-p0])); Shewchuk returns the
// opposite. We negate to match geogram's convention so every existing
// predicate/intersection routine in the codebase keeps working untouched.

namespace prism {
namespace predicates {

// Returns +1 / 0 / -1 with the same sign convention as
// GEO::PCK::orient_3d(p0, p1, p2, p3).
int orient_3d(const double* p0, const double* p1, const double* p2,
              const double* p3);

// Returns +1 / 0 / -1 with the same sign convention as
// GEO::PCK::orient_2d(p0, p1, p2).
int orient_2d(const double* p0, const double* p1, const double* p2);

}  // namespace predicates
}  // namespace prism

#endif
