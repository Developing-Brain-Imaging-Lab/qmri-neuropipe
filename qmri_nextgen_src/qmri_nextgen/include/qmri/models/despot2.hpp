#pragma once
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <cmath>
#include "despot1.hpp"
#include "../core/model_traits.hpp"
#include "../core/linear_solvers.hpp"
#include "../signals/ssfp.hpp"


namespace qmri {

struct DESPOT2Params {
 
  // Own or view your protocol vectors; keep them immutable after construction.
    const Eigen::VectorXf flip;
    const Eigen::VectorXf tr;
    const Eigen::VectorXf te;
    const Eigen::VectorXf phase;

    DESPOT2Params(const Eigen::VectorXf& ssfpFlip, const Eigen::VectorXf& ssfpTR,  const Eigen::VectorXf& ssfpPhases): flip(ssfpFlip*deg2rad), tr(ssfpTR), phase(ssfpPhases) {}

    inline int num_params() const { return 2; }          // {M0, T2, F0}
    inline int num_obs()   const { return static_cast<int>(flip.size()); }
    
    void print(std::ostream & os, int id){

        std::string indent;
        for(int i=0; i<id; i++)
            indent += " ";
        
        os << std::endl;
        os << "Driven Equilibrium Single Pulse Observation of T2" << std::endl;
        os << std::endl;
        os << indent << "SSFP Parameters: " << std::endl;
        os << indent << indent << "TRs (seconds): " << tr.transpose() << std::endl;
        os << indent << indent << "Flip Angles (degrees): " << flip.transpose()/deg2rad << std::endl;
        os << indent << indent << "Phase Increments (degrees): " << phase.transpose() << std::endl;
        os << std::endl;

    }
    
};

struct DESPOT2 {
    
    const DESPOT2Params* params;  // non-owning; safe if base outlives voxel
    const Eigen::VectorXf fixed;
    static constexpr int kNumParams = 2;
    
    DESPOT2(const DESPOT2Params& b, const Eigen::VectorXf f0) : params(&b), fixed(f0) {}
    
    int num_params() const { return 2; }
    int num_obs()   const { return params->num_obs(); }
    
    Eigen::VectorXf lower_bounds() const { return (Eigen::Vector2f() << 0.f, T2_MIN).finished(); }
    Eigen::VectorXf upper_bounds() const { return (Eigen::Vector2f() << std::numeric_limits<float>::infinity(), T2_MAX).finished(); }

    Eigen::VectorXf residuals(const Eigen::VectorXf& p, const Eigen::VectorXf& x) const {
        const float M0 = p[0], T2 = p[1];
        const float T1 = fixed[0], B1 = fixed[1], F0 = fixed[2];
        const int N = num_obs();
        
        Eigen::VectorXf r(N);
        for (int k=0;k<N;++k) { r[k] = ssfp(M0, T1, T2, B1, F0, params->flip[k], params->tr[k], params->phase[k]) - x[k]; }
           
        return r;
    }
    

//    Eigen::MatrixXf jacobian(const Eigen::VectorXf& p, const Eigen::VectorXf&) const {
//        const float M0 = p[0], T1 = p[1], B1 = fixed[0];
//        const int N = num_obs(); const int Pn = num_params();
//        Eigen::MatrixXf J(N, Pn);
//        for (int k = 0; k < N; ++k) {
//              const float alpha = params->flip[k] * float(M_PI/180.0) * B1;
//              const float sa = std::sin(alpha), ca = std::cos(alpha);
//              const float E1 = std::exp(-params->tr[k] / T1);
//              const float denom = 1.f - ca*E1;
//              const float num   = 1.f - E1;
//              const float yhat  = M0 * sa * num / denom;
//
//              // r = x - yhat -> dr/dθ = - dyhat/dθ
//              // ∂yhat/∂M0
//              const float dy_dM0 = sa * num / denom;
//              // ∂yhat/∂T1 (standard SPGR derivatives; omit here for brevity/you can fill exact form)
//              const float dE1dT1 = (params->tr[k] / (T1*T1)) * E1;
//              // One safe approach: finite-diff T1 derivative if you don't want to derive analytic formula now:
//              // but keep analytic if you already have it.
//
//              // Placeholder finite-diff for T1 derivative (small epsilon)
//              const float eps = 1e-3f * std::max(1.f, T1);
//              const float E1p = std::exp(-params->tr[k] / (T1 + eps));
//              const float yhat_p = M0 * sa * (1.f - E1p) / (1.f - ca*E1p);
//              const float dy_dT1 = (yhat_p - yhat) / eps;
//
//              J(k, 0) = -dy_dM0;
//              J(k, 1) = -dy_dT1;
//            }
//            return J;
//    }

    
    struct AutoDiffFunctor {
        const DESPOT2Params* params;
        const Eigen::VectorXf fixed;
        Eigen::VectorXf xobs;

        AutoDiffFunctor(const DESPOT2& m, const Eigen::VectorXf& x): params(m.params), fixed(m.fixed), xobs(x) {}

        template <typename T>
        bool operator()(const T* const x, T* residuals) const {
            const T M0 = x[0];
            const T T2 = x[1];
            
            const T T1 = T(fixed[0]);
            const T B1 = T(fixed[1]);
            const T F0 = T(fixed[2]);
            
            const int N = params->num_obs();
            
            for (int k=0;k<N;++k) { residuals[k] = ssfp(M0, T1, T2, B1, F0, T(params->flip[k]), T(params->tr[k]), T(params->phase[k])) - T(xobs[k]); }

            
            return true;
        }
    };
    using AutoDiffFunctor = AutoDiffFunctor;  // <-- this alias is what our factory looks for

    Eigen::VectorXf linear_solve(const Eigen::Ref<const Eigen::VectorXf>& x,
                                 const Eigen::Ref<const Eigen::VectorXf>& p0,
                                 bool weighted) const {
        
        
        
    }
    
    Eigen::VectorXf initial_guess(const Eigen::Ref<const Eigen::VectorXf>& x,
                                  const Eigen::Ref<const Eigen::VectorXf>& pd) const {
        
        
        
        Eigen::VectorXf x0(2);
        x0 << 10000., std::max(0.045*fixed[0], 1.5*params->tr[0]);
        
        return x0;
    }
    
    
};

struct DESPOT2Binder {
    
    const DESPOT2Params& params;
    const Eigen::MatrixXf& fixed; // size = #voxels (in the flattened/tiled space)

    inline DESPOT2 operator()(int v) const { return DESPOT2(params, fixed.col(v)); }
};

template<> struct ModelTraits<DESPOT2> {
  static constexpr bool HasAnalyticJacobian = false;
  static constexpr int  MaxParams = 2;
};


} // namespace qmri
