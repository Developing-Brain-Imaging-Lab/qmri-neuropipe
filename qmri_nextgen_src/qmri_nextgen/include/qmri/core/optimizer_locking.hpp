#pragma once
#include <Eigen/Dense>
#include <vector>
#include <ceres/ceres.h>            // <-- only need this
#include "optimizer_tags.hpp"
#include "optimizer_bridge.hpp"
#include "optim_src.hpp"

namespace qmri {

inline void _validate_sizes_or_throw(int nr, int np,
                                     const Eigen::VectorXf& x,
                                     const Eigen::VectorXf& p0) {
  if (nr <= 0) throw std::invalid_argument("solve_voxel: num_obs <= 0");
  if (np <= 0) throw std::invalid_argument("solve_voxel: num_params <= 0");
  if (x.size()  != nr) throw std::invalid_argument("solve_voxel: x.size() != num_obs()");
  if (p0.size() != np) throw std::invalid_argument("solve_voxel: p0.size() != num_params()");
}

template<typename Model>
inline Eigen::VectorXf solve_voxel(const Model& m,
                                   const Eigen::VectorXf& x,
                                   const Eigen::VectorXf& p0,
                                   const CeresTag& opts) {
    
    // Infer sizes from model; fall back to vector sizes if model reports 0 for nr
    int nr = m.num_obs();
    int np = m.num_params();
    
    if (nr <= 0) nr = static_cast<int>(x.size());
    _validate_sizes_or_throw(nr, np, x, p0);
    
    //First initialize parameters
    Eigen::VectorXd x0 = m.initial_guess(x, p0).template cast<double>(); //Model specific initialization

    ceres::Problem problem;
    ceres::CostFunction* cost = make_cost_function<Model>(m, x, nr, np);
    problem.AddResidualBlock(cost, new ceres::CauchyLoss(0.5), x0.data());
    
    //Set upper and lower bounds
    for(int i=0; i<np; i++){
        problem.SetParameterLowerBound(x0.data(), i, m.lower_bounds()[i]);
        problem.SetParameterUpperBound(x0.data(), i, m.upper_bounds()[i]);
    }
    
    
    //Set the options
    FLAGS_minloglevel = 2;   // 0:INFO, 1:WARNING, 2:ERROR, 3:FATAL
    FLAGS_logtostderr = 1;
    
    ceres::Solver::Options options;
    options.max_num_iterations = opts.max_iterations;
    options.logging_type = ceres::SILENT;
    options.minimizer_progress_to_stdout = false;
    
    if (opts.xtol > 0.0) options.parameter_tolerance = opts.xtol;
    options.function_tolerance = opts.ftol;
    options.gradient_tolerance = opts.gtol;

    
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
       
    
    return x0.cast<float>();

}

template<typename Model>
inline Eigen::VectorXf solve_voxel(const Model& m,
                                   const Eigen::VectorXf& x,
                                   const Eigen::VectorXf& p0,
                                   const NLoptTag& opts) {
    int nr = m.num_obs();
    int np = m.num_params();
   
    if (nr <= 0) nr = static_cast<int>(x.size());
    _validate_sizes_or_throw(nr, np, x, p0);

    std::vector<double> lb(np, -1e9), ub(np, 1e9);
    (void)nr; // not used by NLopt directly
    
    return p0;
  //return nlopt_solve<Model>(m, x, p0, lb, ub, opts);
    
}

template<typename Model>
inline Eigen::VectorXf solve_voxel(const Model& m,
                                   const Eigen::VectorXf& x,
                                   const Eigen::VectorXf& p0,
                                   const LinearTag& tag) {
    
    return m.linear_solve(x, p0, tag.weighted).template cast<float>();
    
}


    
} // namespace qmri
