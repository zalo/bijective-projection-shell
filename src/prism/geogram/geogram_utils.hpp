#include <prism/common.hpp>

#ifndef PRISM_NO_CGAL
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/logger.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_reorder.h>
#endif

namespace prism::geo {
  // No-op in PRISM_NO_CGAL builds; otherwise initializes geogram once.
  void init_geogram();

#ifndef PRISM_NO_CGAL
  // switch based on F.cols()
  void to_geogram_mesh(const RowMatd& V, const RowMati& F, GEO::Mesh& M);
  void from_geogram_mesh(const GEO::Mesh& M, RowMatd& V, RowMati& T);
#endif
}
