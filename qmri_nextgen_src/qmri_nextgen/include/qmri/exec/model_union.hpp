#pragma once
#include <variant>
#include <vector>
#include <type_traits>
#include <Eigen/Dense>

#include "../models/despot1.hpp"       // DESPOT1Base, DESPOT1Voxel
#include "../models/despot1_hifi.hpp"  // DESPOT1HiFiBase, DESPOT1HiFiVoxel
#include "data_fit.hpp"        // fit_data(binder, X, P, active, chunk, tag, [cb])

namespace qmri::exec {

// --------------------- Binder types ---------------------
using BinderAny = std::variant<DESPOT1Binder, HiFiBinder>;


// ------------------ Utility: num_params ------------------
inline int num_params(const BinderAny& binder) {
  return std::visit([&](auto const& b)->int {
    auto vm = b(0);                // construct a representative voxel model
    return vm.num_params();        // both voxel types expose num_params()
  }, binder);
}

// ------------------ Allocate & seed P --------------------
inline Eigen::MatrixXf alloc_P(const BinderAny& binder, int V) {
  const int C = num_params(binder);
  return Eigen::MatrixXf(C, V);
}

// --------------- Run fit (no progress cb) ----------------
template <typename Tag>
inline void run_fit(const BinderAny& binder,
                    const Eigen::MatrixXf& X_obs,
                    Eigen::MatrixXf& P,
                    const std::vector<int>& active,
                    int chunk,
                    const Tag& tag)
{
  std::visit([&](auto const& b){
    qmri::fit_data(b, X_obs, P, active, chunk, tag);
  }, binder);
}

// ------------- Run fit (with progress cb) ----------------
// ProgressCB signature: void(int done, int total)
template <typename Tag, typename ProgressCB>
inline void run_fit(const BinderAny& binder,
                    const Eigen::MatrixXf& X_obs,
                    Eigen::MatrixXf& P,
                    const std::vector<int>& active,
                    int chunk,
                    const Tag& tag,
                    ProgressCB progress_cb)
{
  std::visit([&](auto const& b){
    qmri::fit_data(b, X_obs, P, active, chunk, tag, progress_cb);
  }, binder);
}

// --------------- Helper: is DESPOT1 or HiFi --------------

enum class FamilyKind { DESPOT1, DESPOT1_HIFI };
inline FamilyKind which_family(const BinderAny& binder) {
  return std::holds_alternative<DESPOT1Binder>(binder) ? FamilyKind::DESPOT1
                                                  : FamilyKind::DESPOT1_HIFI;
}

} // namespace qmri::exec
