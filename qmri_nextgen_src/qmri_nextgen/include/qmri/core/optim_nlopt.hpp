#pragma once
#include "optimizer_tags.hpp"

// Thin overload to keep builds green if nlopt_solve(...) doesn't yet take NLoptTag.
namespace qmri {
//template<typename Model>
//Eigen::VectorXf nlopt_solve(const Model& m,
//                            const Eigen::Ref<const Eigen::VectorXf>& x,
//                            const Eigen::Ref<const Eigen::VectorXf>& p0,
//                            const std::vector<double>& lb,
//                            const std::vector<double>& ub,
//                            const NLoptTag& /*opts*/) {
//  return nlopt_solve<Model>(m, x, p0, lb, ub);
} // namespace qmri
