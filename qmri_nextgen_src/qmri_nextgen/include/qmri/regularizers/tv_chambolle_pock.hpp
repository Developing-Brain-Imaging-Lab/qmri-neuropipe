#pragma once
#include <Eigen/Dense>
#include <cmath>

namespace qmri {
inline void tv_chambolle_pock_regularize(Eigen::Ref<Eigen::MatrixXf> P,
                                         int NX, int NY, int NZ,
                                         float lambda,
                                         int iters = 30,
                                         float tau = 0.02f,
                                         float sigma = 0.02f,
                                         float theta = 1.0f) {
  const int V = NX*NY*NZ;
  const int Pn = P.rows();
  auto idx=[&](int x,int y,int z){ return (z*NY + y)*NX + x; };
  auto grad=[&](const Eigen::VectorXf& u,Eigen::VectorXf& gx,Eigen::VectorXf& gy,Eigen::VectorXf& gz){
    gx.setZero(V); gy.setZero(V); gz.setZero(V);
    for (int z=0; z<NZ; ++z) for (int y=0; y<NY; ++y) for (int x=0; x<NX; ++x){
      int v=idx(x,y,z);
      if (x<NX-1) gx[v]=u[idx(x+1,y,z)]-u[v];
      if (y<NY-1) gy[v]=u[idx(x,y+1,z)]-u[v];
      if (z<NZ-1) gz[v]=u[idx(x,y,z+1)]-u[v];
    }
  };
  auto div=[&](const Eigen::VectorXf& px,const Eigen::VectorXf& py,const Eigen::VectorXf& pz,Eigen::VectorXf& d){
    d.setZero(V);
    for (int z=0; z<NZ; ++z) for (int y=0; y<NY; ++y) for (int x=0; x<NX; ++x){
      int v=idx(x,y,z);
      float dxm=(x>0)? px[idx(x,y,z)]-px[idx(x-1,y,z)] : px[idx(x,y,z)];
      float dym=(y>0)? py[idx(x,y,z)]-py[idx(x,y-1,z)] : py[idx(x,y,z)];
      float dzm=(z>0)? pz[idx(x,y,z)]-pz[idx(x,y,z-1)] : pz[idx(x,y,z)];
      d[v]=dxm+dym+dzm;
    }
  };
  for (int p=0;p<Pn;++p){
    Eigen::VectorXf u=P.row(p).transpose(), u_bar=u;
    Eigen::VectorXf px=Eigen::VectorXf::Zero(V), py=Eigen::VectorXf::Zero(V), pz=Eigen::VectorXf::Zero(V);
    Eigen::VectorXf gx(V),gy(V),gz(V), divp(V);
    for (int k=0;k<iters;++k){
      grad(u_bar,gx,gy,gz);
      px.array() += sigma*gx.array();
      py.array() += sigma*gy.array();
      pz.array() += sigma*gz.array();
      for (int v=0; v<V; ++v){
        float n=std::sqrt(px[v]*px[v]+py[v]*py[v]+pz[v]*pz[v]);
        if (n>lambda && n>0.f){ float s=lambda/n; px[v]*=s; py[v]*=s; pz[v]*=s; }
      }
      div(px,py,pz,divp);
      Eigen::VectorXf u_next = u + tau * divp;
      u_bar = u_next + theta*(u_next - u);
      u = std::move(u_next);
    }
    P.row(p)=u.transpose();
  }
}
} // namespace qmri
