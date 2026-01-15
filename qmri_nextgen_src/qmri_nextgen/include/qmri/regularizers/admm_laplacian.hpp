#pragma once
#include <Eigen/Dense>

namespace qmri {
inline void admm_laplacian_regularize(Eigen::Ref<Eigen::MatrixXf> P,
                                      int NX, int NY, int NZ,
                                      float lambda, int iters=10) {
  const int V = NX*NY*NZ;
  const int Pn = P.rows();
  auto idx=[&](int x,int y,int z){ return (z*NY + y)*NX + x; };
  Eigen::MatrixXf Q = P;
  Eigen::MatrixXf U = Eigen::MatrixXf::Zero(Pn,V);
  for (int k=0;k<iters;++k) {
    Eigen::MatrixXf Pnew = P;
    for (int z=0; z<NZ; ++z) for (int y=0; y<NY; ++y) for (int x=0; x<NX; ++x) {
      int v = idx(x,y,z);
      for (int p=0;p<Pn;++p) {
        float center = P(p,v), sum=0.f; int cnt=0;
        auto add=[&](int xi,int yi,int zi){
          if (xi>=0 && xi<NX && yi>=0 && yi<NY && zi>=0 && zi<NZ) { sum+=P(p,idx(xi,yi,zi)); cnt++; }
        };
        add(x-1,y,z); add(x+1,y,z); add(x,y-1,z); add(x,y+1,z); add(x,y,z-1); add(x,y,z+1);
        float smooth = (cnt? sum/cnt : center);
        float rhs = Q(p,v) - U(p,v);
        float alpha = lambda / (lambda + 1.0f);
        Pnew(p,v) = (1.0f - alpha) * rhs + alpha * smooth;
      }
    }
    P = Pnew; Q = P + U; U += P - Q;
  }
}

inline void admm_laplacian_regularize_masked(Eigen::Ref<Eigen::MatrixXf> P,
                                             int NX, int NY, int NZ,
                                             float lambda,
                                             const Eigen::Ref<const Eigen::VectorXf>& mask,
                                             int iters=10) {
  const int V = NX*NY*NZ;
  const int Pn = P.rows();
  auto idx=[&](int x,int y,int z){ return (z*NY + y)*NX + x; };

  Eigen::MatrixXf Q = P;
  Eigen::MatrixXf U = Eigen::MatrixXf::Zero(Pn,V);

  for (int k=0;k<iters;++k) {
    Eigen::MatrixXf Pnew = P;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int p=0; p<Pn; ++p)
      for (int z=0; z<NZ; ++z)
        for (int y=0; y<NY; ++y)
          for (int x=0; x<NX; ++x) {
            const int v = idx(x,y,z);
            if (mask[v]==0.0f) continue;

            float sum = 0.f; int cnt = 0;
            auto add=[&](int xi,int yi,int zi){
              if (xi>=0 && xi<NX && yi>=0 && yi<NY && zi>=0 && zi<NZ){
                const int vn = idx(xi,yi,zi);
                if (mask[vn]!=0.0f){ sum += P(p,vn); ++cnt; }
              }
            };
            add(x-1,y,z); add(x+1,y,z); add(x,y-1,z); add(x,y+1,z); add(x,y,z-1); add(x,y,z+1);
            const float smooth = (cnt ? sum/cnt : P(p,v));
            const float rhs = Q(p,v) - U(p,v);
            const float alpha = lambda / (lambda + 1.0f);
            Pnew(p,v) = (1.0f - alpha)*rhs + alpha*smooth;
          }

    P.swap(Pnew);
    Q = P + U;
    U += P - Q;
  }
}

} // namespace qmri
