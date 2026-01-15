#pragma once
#include <Eigen/Dense>
#include "../core/optimizer_bridge.hpp"

namespace qmri {
template<typename Model, typename Backend>
void fit_tile(const Model& m,
              const Eigen::Ref<const Eigen::MatrixXf>& X_obs,
              Eigen::Ref<Eigen::MatrixXf> P_io) {
  const int V = static_cast<int>(X_obs.cols());
  #pragma omp parallel for schedule(static)
  for (int v=0; v<V; ++v) {
    auto x = X_obs.col(v);
    auto p0 = P_io.col(v);
    auto p = solve_voxel<Model>(m, x, p0, Backend{});
    P_io.col(v) = p;
  }
}
} // namespace qmri
