#pragma once
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <cmath>
#include "../core/model_traits.hpp"
#include "../core/linear_solvers.hpp"
#include "../signals/spgr.hpp"

namespace qmri {

struct DESPOT1Params {
 
  // Own or view your protocol vectors; keep them immutable after construction.
    const Eigen::VectorXf flip;  // size = Nobs
    const Eigen::VectorXf tr;    // size = Nobs

    DESPOT1Params(const Eigen::VectorXf& flip_deg, const Eigen::VectorXf& TR_ms): flip(flip_deg*deg2rad), tr(TR_ms) {}

    int num_params() const { return 2; }          // {M0, T1}
    int num_obs()   const { return static_cast<int>(flip.size()); }
    
    void print(std::ostream & os, int id){

        std::string indent;
        for(int i=0; i<id; i++)
            indent += " ";
        
        os << std::endl;
        os << "Driven Equilibrium Single Pulse Observation of T1" << std::endl;
        os << std::endl;
        os << indent << "SPGR Parameters: " << std::endl;
        
        os << indent << indent << "TR (seconds): " << tr[0] << std::endl;
        os << indent << indent << "Flip Angles (degrees): " << (flip.transpose()/deg2rad) << std::endl;
        os << std::endl;
    }
    
};

struct DESPOT1 {
    
    const DESPOT1Params* params;  // non-owning; safe if base outlives voxel
    Eigen::VectorXf fixed;
    static constexpr int kNumParams = 2;
    
    DESPOT1(const DESPOT1Params& b, Eigen::VectorXf f0) : params(&b), fixed(f0) {}
    
    int num_params() const { return 2; }
    int num_obs()   const { return params->num_obs(); }
    void set_fixed(Eigen::VectorXf& f){ fixed = f; }
    
    Eigen::VectorXf lower_bounds() const { return (Eigen::Vector2f() << 0.f, T1_MIN).finished(); }
    Eigen::VectorXf upper_bounds() const { return (Eigen::Vector2f() << std::numeric_limits<float>::infinity(), T1_MAX).finished(); }

    Eigen::VectorXf residuals(const Eigen::VectorXf& p, const Eigen::VectorXf& x) const {
        const float M0 = p[0], T1 = p[1], B1 = fixed[0];
        const int N = num_obs();
        
        Eigen::VectorXf r(N);
        for (int i=0;i<N;++i) { r[i] = qmri::spgr(M0, T1, B1, params->flip[i], params->tr[i]) - x[i]; }
        
        return r;
    }
    

    struct AutoDiffFunctor {
        const DESPOT1Params* params;
        const Eigen::VectorXf fixed;
        Eigen::VectorXf xobs;

        AutoDiffFunctor(const DESPOT1& m, const Eigen::VectorXf& x): params(m.params), fixed(m.fixed), xobs(x) {}

        template <typename T>
        bool operator()(const T* const x, T* residuals) const {
            const T M0 = x[0];
            const T T1 = x[1];
            const T B1 = T(fixed[0]);
            const int N = static_cast<int>(params->flip.size());
            
            for (int k = 0; k < N; ++k) { residuals[k] = spgr(M0, T1, B1, T(params->flip[k]), T(params->tr[k])) - T(xobs[k]); }
            
        return true;
      }
    };
    using AutoDiffFunctor = AutoDiffFunctor;  // <-- this alias is what our factory looks for

    Eigen::VectorXf linear_solve(const Eigen::Ref<const Eigen::VectorXf>& x,
                                 const Eigen::Ref<const Eigen::VectorXf>& p0,
                                 bool weighted) const {
        
        const int np = num_params();
        const float B1 = fixed[0];
        
        Eigen::VectorXd pd = p0.template cast<double>();
        Eigen::VectorXd xd = x.template cast<double>();
        
        Eigen::VectorXd sa = sin(B1*params->flip.template cast<double>().array());
        Eigen::VectorXd ca = cos(B1*params->flip.template cast<double>().array());
        Eigen::VectorXd ta = tan(B1*params->flip.template cast<double>().array());
        
        Eigen::VectorXd X = xd.array() / ta.array();
        Eigen::VectorXd Y = xd.array() / sa.array();
        
        Eigen::Vector2d z = Eigen::Vector2d::Zero();
        double m0, t1;
        
        if (weighted){
            
            double e1, new_t1, new_m0;
            double threshold = 0.0001;
            int iters = 10;
            
            Eigen::VectorXd W(X.rows());
            W.setOnes();
            
            z  = weighted_least_squares(X, Y, W);
            m0 = z[1]/(1.00-z[0]);
            t1 = -params->tr[0]/log(z[0]);
            
            for(int its=0; its<iters; its++){
                e1 = exp(-params->tr[0]/t1);
                W = sa.array()/(1.00-e1*ca.array()).square();
                
                z = weighted_least_squares(X, Y, W);
                new_m0 = z[1]/(1.00-z[0]);
                new_t1 = -params->tr[0]/log(z[0]);
                
                if( (abs(new_m0 - m0)<threshold) && (abs(new_t1 - t1)<threshold) ) break;
                else{
                    m0 = new_m0;
                    t1 = new_t1;
                }
            }
        } else {
                
            z  = ordinary_least_squares(X, Y);
            m0 = z[1]/(1.00-z[0]);
            t1 = -params->tr[0]/log(z[0]);
            
        }
        
        pd[0] = m0;
        pd[1] = t1;
        
        return pd.template cast<float>();
        
    }
    
    Eigen::VectorXf initial_guess(const Eigen::Ref<const Eigen::VectorXf>& x,
                                  const Eigen::Ref<const Eigen::VectorXf>& pd) const {
        return this->linear_solve(x, pd, false);
    }
    

        
    Eigen::VectorXf solve_voxel_nlopt(const Eigen::Ref<const Eigen::VectorXf>& x,
                                      const Eigen::Ref<const Eigen::VectorXf>& p0) const {
        // NLOPT-specific fitting code for DESPOT1
    }
    
    
};

struct DESPOT1Binder {
    
    std::shared_ptr<const DESPOT1Params> params;
    std::shared_ptr<const Eigen::MatrixXf> B1map; // size = #voxels (in the flattened/tiled space)
    
    DESPOT1Binder() = default;
    DESPOT1Binder(std::shared_ptr<const qmri::DESPOT1Params> par,
                  std::shared_ptr<const Eigen::MatrixXf> b1): params(std::move(par)), B1map(std::move(b1)) {}

    inline DESPOT1 operator()(int v) const { return { *params, (*B1map).col(v) }; }
};

template<> struct ModelTraits<DESPOT1> {
  static constexpr bool HasAnalyticJacobian = true;
  static constexpr int  MaxParams = 2;
};


} // namespace qmri
