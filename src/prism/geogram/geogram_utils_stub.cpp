// Stub implementation of geogram_utils for PRISM_NO_CGAL/PRISM_WASM builds.
// Mirrors the API the rest of prism_library expects (a single init_geogram()
// entry point) without dragging in geogram itself.

#include "geogram_utils.hpp"

void prism::geo::init_geogram() {}
