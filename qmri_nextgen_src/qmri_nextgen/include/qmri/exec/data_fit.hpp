#pragma once
#include <Eigen/Dense>
#include <type_traits>
#include <vector>
#include <functional>
#include "batch_executor_masked.hpp"
#include "../core/optimizer_tags.hpp"

namespace qmri {

// Overload A: no progress callback → calls non-progress tile executor
template<typename Binder, typename BackendTag>
inline void fit_data(const Binder& binder,
                     const Eigen::Ref<const Eigen::MatrixXf>& X_obs,
                     Eigen::Ref<Eigen::MatrixXf> P_io,
                     const std::vector<int>& active_vox,
                     int chunk_size,
                     const BackendTag& tag){
  (void)chunk_size; // not used in non-progress path
  fit_tile_masked(binder, X_obs, P_io, active_vox, chunk_size, tag);
}

// Overload B: progress callback present → calls progress tile executor
template<typename Binder, typename BackendTag, typename Callback>
inline void fit_data(const Binder& binder,
                     const Eigen::Ref<const Eigen::MatrixXf>& X_obs,
                     Eigen::Ref<Eigen::MatrixXf> P_io,
                     const std::vector<int>& active_vox,
                     int chunk_size,
                     const BackendTag& tag,
                     Callback progress_cb){
    
  fit_tile_progress_masked(binder, X_obs, P_io, active_vox, chunk_size, tag, progress_cb);
    
}

} // namespace qmri
