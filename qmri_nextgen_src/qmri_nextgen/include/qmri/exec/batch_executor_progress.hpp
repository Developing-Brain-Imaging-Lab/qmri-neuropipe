#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include "../core/optimizer_bridge.hpp"

namespace qmri {

// Chunked tile fit with per-chunk progress callback.
// Processes voxels in blocks; inside each block, use OpenMP parallel-for.
// After each block, calls cb(processed_voxels, total_voxels).
template<typename Model, typename Backend, typename Callback>
void fit_tile_progress(const Model& m,
                       const Eigen::Ref<const Eigen::MatrixXf>& X_obs,
                       Eigen::Ref<Eigen::MatrixXf> P_io,
                       int chunk_size,
                       Callback cb) {
  const int V = static_cast<int>(X_obs.cols());
  int processed = 0;
  for (int s = 0; s < V; s += chunk_size) {
    int e = std::min(s + chunk_size, V);
    int n = e - s;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
      int v = s + i;
      auto x  = X_obs.col(v);
      auto p0 = P_io.col(v);
      auto p  = solve_voxel<Model>(m, x, p0, Backend{});
      P_io.col(v) = p;
    }
    processed = e;
    cb(processed, V);
  }
}

} // namespace qmri
