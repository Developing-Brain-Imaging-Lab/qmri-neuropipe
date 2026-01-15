#pragma once
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include "../models/despot1.hpp"
#include "../core/optimizer_locking.hpp"
#include "../core/optimizer_tags.hpp"

namespace qmri {

template<typename Binder, typename BackendTag>
void fit_tile_masked(const Binder& binder,
                     const Eigen::Ref<const Eigen::MatrixXf>& X_obs,
                     Eigen::Ref<Eigen::MatrixXf> P_io,
                     const std::vector<int>& active_vox,
                     int chunk_size,
                     const BackendTag& tag) {
  const int Vsel = static_cast<int>(active_vox.size());
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < Vsel; ++i) {
      int v = active_vox[i];
      const Eigen::VectorXf x  = X_obs.col(v);
      const Eigen::VectorXf p0 = P_io.col(v);
      auto bound = binder(v);
      
      const Eigen::VectorXf p = qmri::solve_voxel(bound, x, p0, tag);
      P_io.col(v) = p;
  }
}
template<typename Binder, typename BackendTag, typename Callback>
void fit_tile_progress_masked(const Binder& binder,
                              const Eigen::Ref<const Eigen::MatrixXf>& X_obs,
                              Eigen::Ref<Eigen::MatrixXf> P_io,
                              const std::vector<int>& active_vox,
                              int chunk_size,
                              const BackendTag& tag,
                              Callback cb) {
  const int Vsel = static_cast<int>(active_vox.size());
  for (int s = 0; s < Vsel; s += chunk_size) {
    int e = std::min(s + chunk_size, Vsel);
    int n = e - s;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int v = active_vox[s + i];
        const Eigen::VectorXf x  = X_obs.col(v);
        const Eigen::VectorXf p0 = P_io.col(v);

        auto bound = binder(v);
        const Eigen::VectorXf p = qmri::solve_voxel(bound, x, p0, tag);
        P_io.col(v) = p;
    }
    cb(e, Vsel);
  }
}
} // namespace qmri
