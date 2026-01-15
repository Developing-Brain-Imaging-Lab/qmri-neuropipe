#pragma once
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <cmath>
#include "despot1.hpp"
#include "../core/model_traits.hpp"
#include "../core/linear_solvers.hpp"
#include "../signals/spgr.hpp"
#include "../signals/irspgr.hpp"


namespace qmri {

struct HiFiParams {
 
  // Own or view your protocol vectors; keep them immutable after construction.
    const DESPOT1Params spgr_params;
    
    const Eigen::VectorXf flip;
    const Eigen::VectorXf tr;
    const Eigen::VectorXf ti;
    const Eigen::VectorXf etl;

    HiFiParams(const Eigen::VectorXf& spgrFlip, const Eigen::VectorXf& spgrTR, const Eigen::VectorXf& irspgrTR, const Eigen::VectorXf& irspgrTI, const Eigen::VectorXf& irspgrFlip, const Eigen::VectorXf& irspgrETL): spgr_params(spgrFlip, spgrTR), tr(irspgrTR), ti(irspgrTI), flip(irspgrFlip*deg2rad), etl(irspgrETL) {}

    inline int num_params() const { return 3; }          // {M0, T1, B1}
    inline int num_spgr() const { return static_cast<int>(spgr_params.num_obs()); }
    inline int num_irspgr() const { return static_cast<int>(ti.size()); }
    inline int num_obs()   const { return num_spgr() + num_irspgr(); }
    
    void print(std::ostream & os, int id){

        std::string indent;
        for(int i=0; i<id; i++)
            indent += " ";
        
        os << std::endl;
        os << "Driven Equilibrium Single Pulse Observation of T1 – High-Speed Incorporation of RF Field Inhomogenities (DESPOT1-HIFI)" << std::endl;
        os << std::endl;
        os << indent << "SPGR Parameters: " << std::endl;
        os << indent << indent << "TRs (seconds): " << spgr_params.tr.transpose() << std::endl;
        os << indent << indent << "Flip Angles (degrees): " << spgr_params.flip.transpose() << std::endl;
        os << std::endl;
        
        os << indent << "IR-SPGR Parameters: " << std::endl;
        os << indent << indent << "TRs (seconds): " << tr.transpose() << std::endl;
        os << indent << indent << "TIs (seconds): " << ti.transpose() << std::endl;
        os << indent << indent << "Flip Angles (degrees): " << (flip.transpose()/deg2rad) << std::endl;
        os << std::endl;
    }
    
};

struct DESPOT1HiFi {
    
    const HiFiParams* params;  // non-owning; safe if base outlives voxel
    static constexpr int kNumParams = 3;
    
    DESPOT1HiFi(const HiFiParams& b) : params(&b) {}
    
    int num_params() const { return 3; }
    int num_obs()   const { return params->num_obs(); }
    
    Eigen::VectorXf lower_bounds() const { return (Eigen::Vector3f() << 0.f, T1_MIN, 0.3).finished(); }
    Eigen::VectorXf upper_bounds() const { return (Eigen::Vector3f() << std::numeric_limits<float>::infinity(), T1_MAX, 1.8).finished(); }

    Eigen::VectorXf residuals(const Eigen::VectorXf& p, const Eigen::VectorXf& x) const {
        const float M0 = p[0], T1 = p[1], B1 = p[2];
        const int N = num_obs();
        
        int idx = 0;
        Eigen::VectorXf r(N);
        
        const int Ns = params->num_spgr();
        for (int k=0;k<Ns;++k,++idx) { r[idx] = spgr(M0, T1, B1, params->spgr_params.flip[k], params->spgr_params.tr[k]) - x[idx]; }
           
        const int Ni = params->num_irspgr();
        for (int k=0;k<Ni;++k,++idx){ r[idx] = irspgr(M0, T1, B1, params->flip[k], params->tr[k], params->ti[k], params->etl[k]) - x[idx]; }
            
        return r;
    }
        
    struct AutoDiffFunctor {
        const HiFiParams* params;
        Eigen::VectorXf xobs;

        AutoDiffFunctor(const DESPOT1HiFi& m, const Eigen::VectorXf& x): params(m.params), xobs(x) {}

        template <typename T>
        bool operator()(const T* const x, T* residuals) const {
            const T M0 = x[0];
            const T T1 = x[1];
            const T B1 = x[2];
            
            int idx = 0;
            
            const int Ns = static_cast<int>(params->num_spgr());
            for (int k=0;k<Ns;++k,++idx) { residuals[idx] = spgr(M0, T1, B1, T(params->spgr_params.flip[k]), T(params->spgr_params.tr[k])) - T(xobs[idx]); }
               
            const int Ni = params->num_irspgr();
            for (int k=0;k<Ni;++k,++idx){ residuals[idx] = irspgr(M0, T1, B1, T(params->flip[k]), T(params->tr[k]), T(params->ti[k]), T(params->etl[k])) - T(xobs[idx]); }
            
            return true;
        }
    };
    using AutoDiffFunctor = AutoDiffFunctor;  // <-- this alias is what our factory looks for

    Eigen::VectorXf linear_solve(const Eigen::Ref<const Eigen::VectorXf>& x,
                                 const Eigen::Ref<const Eigen::VectorXf>& p0,
                                 bool weighted) const {
        
        const int np = num_params();
        Eigen::Vector3d pd;
    
        double ax, cx, bx, x0, x1, x2, x3, fax, fbx, fcx, f1, f2;
        double R = 0.61803399, C=1.-R, Precision=0.001;
        
        Eigen::VectorXf d1(3), f0(1);
        Eigen::VectorXf xspgr = x.head(params->num_spgr());
        DESPOT1 despot1(params->spgr_params, f0);
        
        ax = this->lower_bounds()[2];
        cx = this->upper_bounds()[2];
        
        f0[0] = ax;
        despot1.set_fixed(f0);
        d1  = despot1.linear_solve(xspgr, p0, false);
        d1[2] = f0[0];
        fax = this->residuals(d1, x).array().square().sum();
        
        f0[0] = cx;
        despot1.set_fixed(f0);
        d1 = despot1.linear_solve(xspgr, p0, false);
        d1[2] = f0[0];
        fcx = this->residuals(d1, x).array().square().sum();
        
        if (fax < fcx)
            bx = ax + 0.2;
        else
            bx = cx - 0.2;
        
        f0[0] = bx;
        despot1.set_fixed(f0);
        d1 = despot1.linear_solve(xspgr, p0, false);
        d1[2] = f0[0];
        fbx = this->residuals(d1, x).array().square().sum();
        
        x0 = ax;
        x3 = cx;
        
        if ( abs(cx-bx) > abs(bx-ax) ){
            x1 = bx;
            x2 = bx + C*(cx-bx);
        }else{
            x1 = bx;
            x2 = bx - C*(bx-ax);
        }
        
        f0[0] = x1;
        despot1.set_fixed(f0);
        d1 = despot1.linear_solve(xspgr, p0, false);
        d1[2] = f0[0];
        f1 = this->residuals(d1, x).array().square().sum();
        
        f0[0] = x2;
        despot1.set_fixed(f0);
        d1 = despot1.linear_solve(xspgr, p0, false);
        d1[2] = f0[0];
        f2 = this->residuals(d1, x).array().square().sum();
        
        while (abs(x3-x0) > Precision*(abs(x1)+abs(x2))){
            if (f2 < f1){
                x0 = x1;
                x1 = x2;
                x2 = R*x1 + C*x3;
                f1 = f2;
                
                f0[0] = x2;
                despot1.set_fixed(f0);
                d1 = despot1.linear_solve(xspgr, p0, false);
                d1[2] = f0[0];
                f2 = this->residuals(d1, x).array().square().sum();
                
            }else{
                x3 = x2;
                x2 = x1;
                x1  = R*x2 + C*x0;
                f2 = f1;
                
                f0[0] = x1;
                despot1.set_fixed(f0);
                d1 = despot1.linear_solve(xspgr, p0, false);
                d1[2] = f0[0];
                f1 = this->residuals(d1, x).array().square().sum();
            }
        }
        
        if (f1 < f2)
            f0[0] = x1;
        else
            f0[0] = x2;
        
        despot1.set_fixed(f0);
        d1 = despot1.linear_solve(xspgr, p0, false);
        
        pd[0] = d1[0];
        pd[1] = d1[1];
        pd[2] = f0[0];
        
        
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

struct HiFiBinder {
    
    std::shared_ptr<const HiFiParams> params;
    
    HiFiBinder() = default;
    HiFiBinder(std::shared_ptr<const qmri::HiFiParams> par) : params(std::move(par)) {}
   
    inline DESPOT1HiFi operator()(int v) const { return { *params }; }
};

template<> struct ModelTraits<DESPOT1HiFi> {
  static constexpr bool HasAnalyticJacobian = false;
  static constexpr int  MaxParams = 3;
};


} // namespace qmri
