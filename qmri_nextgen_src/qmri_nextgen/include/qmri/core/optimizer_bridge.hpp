#pragma once
#include <Eigen/Dense>
#include <type_traits>
#include <stdexcept>
#include <ceres/ceres.h>
#include <ceres/dynamic_autodiff_cost_function.h>
#include "optimizer_tags.hpp"

namespace qmri {

// ===================== Traits / detectors =====================

// Detect model.jacobian(p, x)
template <typename, typename = void>
struct has_jacobian_method : std::false_type {};
template <typename T>
struct has_jacobian_method<T, std::void_t<
    decltype(std::declval<T>().jacobian(
      std::declval<const Eigen::VectorXf&>(),
      std::declval<const Eigen::VectorXf&>()))>> : std::true_type {};

// Detect optional static flags in the model itself
template <typename, typename = void>
struct has_analytic_flag : std::false_type {};
template <typename T>
struct has_analytic_flag<T, std::void_t<decltype(T::HasAnalyticJacobian)>>
  : std::bool_constant<(bool)T::HasAnalyticJacobian> {};
template <typename, typename = void>
struct has_k_analytic_flag : std::false_type {};
template <typename T>
struct has_k_analytic_flag<T, std::void_t<decltype(T::kHasAnalyticJacobian)>>
  : std::bool_constant<(bool)T::kHasAnalyticJacobian> {};

template <typename Model>
inline constexpr bool kModelHasAnalytic =
    has_jacobian_method<Model>::value
 || has_analytic_flag<Model>::value
 || has_k_analytic_flag<Model>::value;

template <typename, typename = void>
struct static_num_params : std::integral_constant<int, -1> {};

template <typename T>
struct static_num_params<T, std::void_t<decltype(T::kNumParams)>>
  : std::integral_constant<int, T::kNumParams> {};


// Detect presence of Model::AutoDiffFunctor
template <typename, typename = void>
struct has_autodiff_functor : std::false_type {};
template <typename T>
struct has_autodiff_functor<T, std::void_t<typename T::AutoDiffFunctor>> : std::true_type {};

// ===================== Cost functors =====================
template<typename Model>
struct CeresCost : ceres::CostFunction {
  const Model model;
  const Eigen::VectorXf x;
    
  CeresCost(const Model& m, const Eigen::VectorXf& x_) : model(m), x(x_) {
    this->set_num_residuals(model.num_obs());
    this->mutable_parameter_block_sizes()->clear();
    this->mutable_parameter_block_sizes()->push_back(model.num_params());
  }
    
  bool Evaluate(double const* const* params, double* residuals, double** jacobians) const override {
    const int np = model.num_params();
    const int nr = model.num_obs();
    Eigen::Map<const Eigen::VectorXd> pd(params[0], np);
    Eigen::VectorXf pf = pd.cast<float>();
    auto r = model.residuals(pf, x);
    Eigen::Map<Eigen::VectorXd>(residuals, nr) = r.template cast<double>();
      
//    if (jacobians && jacobians[0]) {
//      if constexpr (ModelTraits<Model>::HasAnalyticJacobian) {
//        auto J = model.jacobian(pf, x);
//        Eigen::Map<Eigen::MatrixXd>(jacobians[0], nr, np) = J.template cast<double>();
//      } else {
//        return false;
//      }
//    }
    return true;
  }
};


// Numeric-diff wrapper around residuals
template <typename Model>
struct NumericResidualFunctor {
    
  NumericResidualFunctor(const Model& m, const Eigen::VectorXf& x_, int nr, int np)
    : model(m), x(x_), nr_(nr), np_(np) {}
    
  bool operator()(const double* const params, double* residuals) const {
    Eigen::VectorXf p = Eigen::Map<const Eigen::VectorXd>(params, np_).cast<float>();
    Eigen::VectorXf r = model.residuals(p, x);
    Eigen::Map<Eigen::VectorXd>(residuals, nr_) = r.cast<double>();
    return true;
  }
    
  const Model model;
  const Eigen::VectorXf x;
  const int nr_, np_;
    
};

// ===================== Factory: AutoDiff → Analytic → Numeric =====================

//// Overload with explicit sizes so callers can force nr/np if the model reports 0.
template <typename Model>
inline ceres::CostFunction* make_cost_function(const Model& m,
                                               const Eigen::VectorXf& x,
                                               int nr, int np) {
    if (nr <= 0 || np <= 0) {
        throw std::invalid_argument("make_cost_function: num_obs/num_params must be > 0");
    }

    if constexpr (has_autodiff_functor<Model>::value) {
        
        using Fun = typename Model::AutoDiffFunctor;
        constexpr int K = static_num_params<Model>::value;
        
        if constexpr (K > 0) {
            // compile-time parameter block size (e.g., 2 for DESPOT1)
            return new ceres::AutoDiffCostFunction<Fun, ceres::DYNAMIC, K>(new Fun(m, x), static_cast<int>(nr));
        } else {
            // fallback: dynamic auto-diff
            auto* fun  = new Fun(m, x);
            auto* cost = new ceres::DynamicAutoDiffCostFunction<Fun, 1>(fun);
            cost->AddParameterBlock(static_cast<int>(np));
            cost->SetNumResiduals(static_cast<int>(nr));
            return cost;
        }
    } else if constexpr (kModelHasAnalytic<Model>) { return new CeresCost<Model>(m, x); }
        
    else {
        using Fn = NumericResidualFunctor<Model>;
        return new ceres::NumericDiffCostFunction<Fn, ceres::CENTRAL, ceres::DYNAMIC, ceres::DYNAMIC>(new Fn(m, x, nr, np), ceres::TAKE_OWNERSHIP, nr, np);
    }
}

// Backward-compatible overload that queries the model first
template <typename Model>
inline ceres::CostFunction* make_cost_function(const Model& m,
                                               const Eigen::VectorXf& x) {
  const int nr0 = m.num_obs();
  const int np0 = m.num_params();
  const int nr  = (nr0 > 0) ? nr0 : static_cast<int>(x.size());
  if (np0 <= 0) {
    throw std::invalid_argument("make_cost_function(model,x): model.num_params() must be > 0");
  }
  return make_cost_function<Model>(m, x, nr, np0);
}

} // namespace qmri
