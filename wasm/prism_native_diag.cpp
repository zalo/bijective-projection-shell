// Native head-to-head diagnostic. Reads an OBJ, runs PrismCage::PrismCage
// (V, F, doosabineps, initial_step, kSurface), and prints the same per-vertex
// thickness statistics that the WASM demo's heatmap reports.
//
// Built TWICE:
//   - Without PRISM_WASM (default native): full CGAL + geogram path.
//   - With PRISM_WASM=ON: Shewchuk + libigl AABB + spatial-hash path.
// Compare the two outputs on the identical input to verify the swap-out
// hasn't introduced a regression.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <prism/PrismCage.hpp>
#include <prism/common.hpp>
#include <prism/geogram/AABB.hpp>

static bool read_obj(const std::string& path, RowMatd& V, RowMati& F) {
  std::ifstream in(path);
  if (!in) return false;
  std::vector<double> verts;
  std::vector<int> faces;
  std::string line;
  while (std::getline(in, line)) {
    if (line.size() < 2) continue;
    std::istringstream ss(line);
    std::string tag;
    ss >> tag;
    if (tag == "v") {
      double x, y, z;
      ss >> x >> y >> z;
      verts.insert(verts.end(), {x, y, z});
    } else if (tag == "f") {
      int a, b, c;
      ss >> a >> b >> c;
      faces.insert(faces.end(), {a - 1, b - 1, c - 1});
    }
  }
  size_t nV = verts.size() / 3, nF = faces.size() / 3;
  V.resize(nV, 3);
  F.resize(nF, 3);
  for (size_t i = 0; i < nV; i++)
    V.row(i) << verts[3 * i], verts[3 * i + 1], verts[3 * i + 2];
  for (size_t i = 0; i < nF; i++)
    F.row(i) << faces[3 * i], faces[3 * i + 1], faces[3 * i + 2];
  return true;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <input.obj> [doosabineps=0.2] [initial_step=0.035]\n",
                 argv[0]);
    return 1;
  }
  std::string path = argv[1];
  double doosabineps = argc > 2 ? std::atof(argv[2]) : 0.2;
  double initial_step = argc > 3 ? std::atof(argv[3]) : 0.035;

  RowMatd V;
  RowMati F;
  if (!read_obj(path, V, F)) {
    std::fprintf(stderr, "failed to read %s\n", path.c_str());
    return 1;
  }
  std::fprintf(stderr, "read %lld V, %lld F from %s\n",
               (long long)V.rows(), (long long)F.rows(), path.c_str());
  std::fprintf(stderr, "doosabineps=%g  initial_step=%g\n",
               doosabineps, initial_step);

  auto t0 = std::chrono::steady_clock::now();
  PrismCage cage(V, F, doosabineps, initial_step,
                 PrismCage::SeparateType::kSurface);
  auto t1 = std::chrono::steady_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Per-vertex thickness = ||top - base||.
  size_t nV = cage.mid.size();
  double tmin = std::numeric_limits<double>::infinity(), tmax = 0.0,
         tsum = 0.0;
  size_t zeros = 0;
  // Histogram: 10 bins from 0 to 2*initial_step (the theoretical max).
  const size_t kBins = 10;
  size_t hist[kBins] = {};
  double tmax_theory = 2.0 * initial_step;
  for (size_t i = 0; i < nV; i++) {
    Vec3d d = cage.top[i] - cage.base[i];
    double t = std::sqrt(d.dot(d));
    tmin = std::min(tmin, t);
    tmax = std::max(tmax, t);
    tsum += t;
    if (t < 1e-6) zeros++;
    int b = std::min((int)(t / tmax_theory * kBins), (int)kBins - 1);
    if (b < 0) b = 0;
    hist[b]++;
  }
  double tmean = tsum / nV;

  std::printf(
      "cage V=%zu F=%zu  build=%.0f ms  thickness min=%.3e mean=%.4f max=%.4f "
      "zero=%zu\n",
      nV, cage.F.size(), elapsed_ms, tmin, tmean, tmax, zeros);
  std::printf("histogram (%zu bins, 0 .. %.4f):\n", kBins, tmax_theory);
  for (size_t b = 0; b < kBins; b++) {
    double lo = (double)b / kBins * tmax_theory;
    double hi = (double)(b + 1) / kBins * tmax_theory;
    std::printf("  [%.4f .. %.4f]  %zu  ", lo, hi, hist[b]);
    int bar = (int)std::round(40.0 * hist[b] / nV);
    for (int i = 0; i < bar; i++) std::putchar('#');
    std::putchar('\n');
  }
  return 0;
}
