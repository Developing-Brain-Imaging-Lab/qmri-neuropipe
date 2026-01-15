#pragma once
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <cmath>
#include "despot1.hpp"
#include "../core/model_traits.hpp"
#include "../core/linear_solvers.hpp"
#include "../signals/ssfp.hpp"


namespace qmri {

struct DESPOT2FMParams {
 
  // Own or view your protocol vectors; keep them immutable after construction.
    const Eigen::VectorXf flip;
    const Eigen::VectorXf tr;
    const Eigen::VectorXf te;
    const Eigen::VectorXf phase;

    DESPOT2FMParams(const Eigen::VectorXf& ssfpFlip, const Eigen::VectorXf& ssfpTR,  const Eigen::VectorXf& ssfpPhases): flip(ssfpFlip*deg2rad), tr(ssfpTR), phase(ssfpPhases) {}

    inline int num_params() const { return 3; }          // {M0, T2, F0}
    inline int num_obs()   const { return static_cast<int>(flip.size()); }
    
    void print(std::ostream & os, int id){

        std::string indent;
        for(int i=0; i<id; i++)
            indent += " ";
        
        os << std::endl;
        os << "Driven Equilibrium Single Pulse Observation of T2 - Full Modeling" << std::endl;
        os << std::endl;
        os << indent << "SSFP Parameters: " << std::endl;
        os << indent << indent << "TRs (seconds): " << tr.transpose() << std::endl;
        os << indent << indent << "Flip Angles (degrees): " << flip.transpose()/deg2rad << std::endl;
        os << indent << indent << "Phase Increments (degrees): " << phase.transpose() << std::endl;
        os << std::endl;

    }
    
};

struct DESPOT2FM {
    
    const DESPOT2FMParams* params;  // non-owning; safe if base outlives voxel
    const Eigen::VectorXf fixed;
    static constexpr int kNumParams = 3;
    
    DESPOT2FM(const DESPOT2FMParams& b, const Eigen::VectorXf f0) : params(&b), fixed(f0) {}
    
    int num_params() const { return 3; }
    int num_obs()   const { return params->num_obs(); }
    
    Eigen::VectorXf lower_bounds() const { return (Eigen::Vector3f() << 0.f, T2_MIN, 1e-6).finished(); }
    Eigen::VectorXf upper_bounds() const { return (Eigen::Vector3f() << 100000., T2_MAX, 0.5/params->tr[0] ).finished(); }

    Eigen::VectorXf residuals(const Eigen::VectorXf& p, const Eigen::VectorXf& x) const {
        const float M0 = p[0], T2 = p[1], F0 = p[2];
        const float T1 = fixed[0], B1 = fixed[1];
        const int N = num_obs();
        
        Eigen::VectorXf r(N);
    
        for (int k=0;k<N;++k) { r[k] = ssfp(M0, T1, T2, B1, F0, params->flip[k], params->tr[k], params->phase[k]) - x[k]; }
           
        return r;
    }
    
    
    struct AutoDiffFunctor {
        const DESPOT2FMParams* params;
        const Eigen::VectorXf fixed;
        Eigen::VectorXf xobs;

        AutoDiffFunctor(const DESPOT2FM& m, const Eigen::VectorXf& x): params(m.params), fixed(m.fixed), xobs(x) {}

        template <typename T>
        bool operator()(const T* const x, T* residuals) const {
            const T M0 = x[0];
            const T T2 = x[1];
            const T F0 = x[2];
            
            const T T1 = T(fixed[0]);
            const T B1 = T(fixed[1]);
            
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
        
        Eigen::VectorXf x0(3);
        x0 << 10000., std::max(0.045*fixed[0], 1.5*params->tr[0]), 0.1/params->tr[0];
        
        return x0;
    }
    
    
};

struct DESPOT2FMBinder {
    
    const DESPOT2FMParams& params;
    const Eigen::MatrixXf& fixed; // size = #voxels (in the flattened/tiled space)

    inline DESPOT2FM operator()(int v) const { return DESPOT2FM(params, fixed.col(v)); }
};

template<> struct ModelTraits<DESPOT2FM> {
  static constexpr bool HasAnalyticJacobian = false;
  static constexpr int  MaxParams = 3;
};


} // namespace qmri
