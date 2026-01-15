#pragma once
#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace qmri {

// --- Helper: pseudo-Huber projection radius ---
inline void project_l2_with_huber(float& px, float& py, float& pz, float radius, float huber_eps){
  float nrm2 = px*px + py*py + pz*pz;
  float nrm = std::sqrt(nrm2 + (huber_eps>0.f ? huber_eps*huber_eps : 0.f));
  if (nrm > radius && nrm > 0.f){
    float s = radius / nrm;
    px *= s; py *= s; pz *= s;
  }
}

// ROF TV (scalar): minimize (mu/2)||U - U0||^2 + lambda * TV(U)
inline void tv_rof_scalar_cp(Eigen::Ref<Eigen::MatrixXf> P, // [C x V]
                             const Eigen::Ref<const Eigen::MatrixXf>& P0,
                             int NX, int NY, int NZ,
                             float lambda, float mu,
                             int iters = 30,
                             float tau = 0.02f,
                             float sigma = 0.02f,
                             float theta = 1.0f,
                             float huber_eps = 0.0f) {
  const int V = NX*NY*NZ;
  const int C = P.rows();
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
  for (int c=0;c<C;++c){
    Eigen::VectorXf u = P.row(c).transpose();
    Eigen::VectorXf u0= P0.row(c).transpose();
    Eigen::VectorXf u_bar = u;
    Eigen::VectorXf px = Eigen::VectorXf::Zero(V);
    Eigen::VectorXf py = Eigen::VectorXf::Zero(V);
    Eigen::VectorXf pz = Eigen::VectorXf::Zero(V);
    Eigen::VectorXf gx(V),gy(V),gz(V), divp(V);
    for (int k=0;k<iters;++k){
      grad(u_bar,gx,gy,gz);
      px.array() += sigma*gx.array();
      py.array() += sigma*gy.array();
      pz.array() += sigma*gz.array();
      for (int v=0; v<V; ++v) project_l2_with_huber(px[v],py[v],pz[v], lambda, huber_eps);
      div(px,py,pz,divp);
      Eigen::VectorXf u_next = (u + tau*divp + tau*mu*u0) / (1.0f + tau*mu);
      u_bar = u_next + theta*(u_next - u);
      u = std::move(u_next);
    }
    P.row(c) = u.transpose();
  }
}

// ROF TV (vectorial): joint projection across channels at each voxel
inline void tv_rof_vectorial_cp(Eigen::Ref<Eigen::MatrixXf> P, // [C x V]
                                const Eigen::Ref<const Eigen::MatrixXf>& P0,
                                int NX, int NY, int NZ,
                                float lambda, float mu,
                                int iters = 30,
                                float tau = 0.02f,
                                float sigma = 0.02f,
                                float theta = 1.0f,
                                float huber_eps = 0.0f) {
  const int V = NX*NY*NZ;
  const int C = P.rows();
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
  auto div_chan=[&](int c,const Eigen::VectorXf& px,const Eigen::VectorXf& py,const Eigen::VectorXf& pz,Eigen::VectorXf& d){
    d.setZero(V);
    for (int z=0; z<NZ; ++z) for (int y=0; y<NY; ++y) for (int x=0; x<NX; ++x){
      int v=idx(x,y,z);
      float dxm=(x>0)? px[idx(x,y,z)]-px[idx(x-1,y,z)] : px[idx(x,y,z)];
      float dym=(y>0)? py[idx(x,y,z)]-py[idx(x,y-1,z)] : py[idx(x,y,z)];
      float dzm=(z>0)? pz[idx(x,y,z)]-pz[idx(x,y,z-1)] : pz[idx(x,y,z)];
      d[v]=dxm+dym+dzm;
    }
  };
  std::vector<Eigen::VectorXf> u(C), u0(C), u_bar(C);
  std::vector<Eigen::VectorXf> px(C), py(C), pz(C);
  std::vector<Eigen::VectorXf> gx(C), gy(C), gz(C);
  for (int c=0;c<C;++c){
    u[c]     = P.row(c).transpose();
    u0[c]    = P0.row(c).transpose();
    u_bar[c] = u[c];
    px[c] = Eigen::VectorXf::Zero(V);
    py[c] = Eigen::VectorXf::Zero(V);
    pz[c] = Eigen::VectorXf::Zero(V);
    gx[c].resize(V); gy[c].resize(V); gz[c].resize(V);
  }
  for (int k=0;k<iters;++k){
    for (int c=0;c<C;++c){
      grad(u_bar[c], gx[c], gy[c], gz[c]);
      px[c].array() += sigma * gx[c].array();
      py[c].array() += sigma * gy[c].array();
      pz[c].array() += sigma * gz[c].array();
    }
    // joint projection per voxel
    for (int v=0; v<V; ++v){
      double n2=0.0;
      for (int c=0;c<C;++c){
        n2 += px[c][v]*px[c][v] + py[c][v]*py[c][v] + pz[c][v]*pz[c][v];
      }
      double n = std::sqrt(n2 + (huber_eps>0.f ? huber_eps*huber_eps : 0.0));
      if (n > lambda && n > 0.0){
        float s = static_cast<float>(lambda / n);
        for (int c=0;c<C;++c){ px[c][v]*=s; py[c][v]*=s; pz[c][v]*=s; }
      }
    }
    // primal updates
    for (int c=0;c<C;++c){
      Eigen::VectorXf divp(V);
      div_chan(c, px[c], py[c], pz[c], divp);
      Eigen::VectorXf u_next = (u[c] + tau*divp + tau*mu*u0[c]) / (1.0f + tau*mu);
      u_bar[c] = u_next + theta*(u_next - u[c]);
      u[c] = std::move(u_next);
    }
  }
  for (int c=0;c<C;++c) P.row(c) = u[c].transpose();
}

// -------- Masked variants (only update voxels where mask[v] != 0) --------
inline void tv_rof_scalar_cp_masked(Eigen::Ref<Eigen::MatrixXf> P, // [C x V]
                                    const Eigen::Ref<const Eigen::MatrixXf>& P0,
                                    int NX, int NY, int NZ,
                                    float lambda, float mu,
                                    const Eigen::Ref<const Eigen::VectorXf>& mask,
                                    int iters = 30,
                                    float tau = 0.02f,
                                    float sigma = 0.02f,
                                    float theta = 1.0f,
                                    float huber_eps = 0.0f) {
  const int V = NX*NY*NZ;
  const int C = P.rows();
  auto idx=[&](int x,int y,int z){ return (z*NY + y)*NX + x; };

  std::vector<Eigen::VectorXf> u(C), u_bar(C), u0(C), px(C), py(C), pz(C);
  for (int c=0;c<C;++c){
    u[c]     = P.row(c).transpose();
    u_bar[c] = u[c];
    u0[c]    = P0.row(c).transpose();
    px[c].setZero(V); py[c].setZero(V); pz[c].setZero(V);
  }

  for (int k=0;k<iters;++k){
    // ---- Dual ascent + projection (per-channel, per-voxel) ----
    for (int c=0;c<C;++c){
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
      for (int z=0; z<NZ; ++z)
        for (int y=0; y<NY; ++y)
          for (int x=0; x<NX; ++x) {
            const int v = idx(x,y,z);
            if (mask[v] == 0.0f) { px[c][v]=py[c][v]=pz[c][v]=0.0f; continue; }

            // masked forward differences
            float gx=0.f, gy=0.f, gz=0.f;
            if (x < NX-1) { int vn=idx(x+1,y,z); if (mask[vn]!=0.0f) gx = u_bar[c][vn]-u_bar[c][v]; }
            if (y < NY-1) { int vn=idx(x,y+1,z); if (mask[vn]!=0.0f) gy = u_bar[c][vn]-u_bar[c][v]; }
            if (z < NZ-1) { int vn=idx(x,y,z+1); if (mask[vn]!=0.0f) gz = u_bar[c][vn]-u_bar[c][v]; }

            float pxv = px[c][v] + sigma*gx;
            float pyv = py[c][v] + sigma*gy;
            float pzv = pz[c][v] + sigma*gz;
            project_l2_with_huber(pxv, pyv, pzv, lambda, huber_eps);
            px[c][v]=pxv; py[c][v]=pyv; pz[c][v]=pzv;
          }
    }

    // ---- Primal update (divergence + proximal to u0) ----
    for (int c=0;c<C;++c){
      Eigen::VectorXf u_next = u[c];
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
      for (int z=0; z<NZ; ++z)
        for (int y=0; y<NY; ++y)
          for (int x=0; x<NX; ++x){
            const int v = idx(x,y,z);
            if (mask[v]==0.0f) continue;

            // masked backward divergence
            float dxm = px[c][v];
            float dym = py[c][v];
            float dzm = pz[c][v];
            if (x>0)   { int vm=idx(x-1,y,z); if (mask[vm]!=0.0f) dxm -= px[c][vm]; }
            if (y>0)   { int vm=idx(x,y-1,z); if (mask[vm]!=0.0f) dym -= py[c][vm]; }
            if (z>0)   { int vm=idx(x,y,z-1); if (mask[vm]!=0.0f) dzm -= pz[c][vm]; }
            const float divp = dxm+dym+dzm;

            u_next[v] = (u[c][v] + tau*divp + tau*mu*u0[c][v]) / (1.0f + tau*mu);
          }
      u_bar[c] = u_next + theta*(u_next - u[c]);
      u[c].swap(u_next);
    }
  }

  for (int c=0;c<C;++c)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int v=0; v<V; ++v)
      if (mask[v]!=0.0f) P(c,v) = u[c][v];
}

inline void tv_rof_vectorial_cp_masked(Eigen::Ref<Eigen::MatrixXf> P, // [C x V]
                                       const Eigen::Ref<const Eigen::MatrixXf>& P0,
                                       int NX, int NY, int NZ,
                                       float lambda, float mu,
                                       const Eigen::Ref<const Eigen::VectorXf>& mask,
                                       int iters = 30,
                                       float tau = 0.02f,
                                       float sigma = 0.02f,
                                       float theta = 1.0f,
                                       float huber_eps = 0.0f) {
  const int V = NX*NY*NZ;
  const int C = P.rows();
  auto idx=[&](int x,int y,int z){ return (z*NY + y)*NX + x; };

  std::vector<Eigen::VectorXf> u(C), u_bar(C), u0(C), px(C), py(C), pz(C);
  for (int c=0;c<C;++c){
    u[c]     = P.row(c).transpose();
    u_bar[c] = u[c];
    u0[c]    = P0.row(c).transpose();
    px[c].setZero(V); py[c].setZero(V); pz[c].setZero(V);
  }

  for (int k=0;k<iters;++k){
    // ---- Dual ascent + joint projection (per voxel) ----
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
    for (int z=0; z<NZ; ++z)
      for (int y=0; y<NY; ++y)
        for (int x=0; x<NX; ++x){
          const int v = idx(x,y,z);
          if (mask[v]==0.0f) {
            for (int c=0;c<C;++c){ px[c][v]=py[c][v]=pz[c][v]=0.0f; }
            continue;
          }

          // compute forward diffs for each channel
          float n2 = 0.f;
          for (int c=0;c<C;++c){
            float gx=0.f, gy=0.f, gz=0.f;
            if (x < NX-1) { int vn=idx(x+1,y,z); if (mask[vn]!=0.0f) gx = u_bar[c][vn]-u_bar[c][v]; }
            if (y < NY-1) { int vn=idx(x,y+1,z); if (mask[vn]!=0.0f) gy = u_bar[c][vn]-u_bar[c][v]; }
            if (z < NZ-1) { int vn=idx(x,y,z+1); if (mask[vn]!=0.0f) gz = u_bar[c][vn]-u_bar[c][v]; }
            px[c][v] += sigma*gx;
            py[c][v] += sigma*gy;
            pz[c][v] += sigma*gz;
            n2 += px[c][v]*px[c][v] + py[c][v]*py[c][v] + pz[c][v]*pz[c][v];
          }

          // joint (vectorial) shrink / projection
          float eps2 = (huber_eps>0.f) ? huber_eps*huber_eps : 0.f;
          float nrm  = std::sqrt(n2 + eps2);
          if (nrm > lambda && nrm > 0.f){
            float s = lambda / nrm;
            for (int c=0;c<C;++c){ px[c][v]*=s; py[c][v]*=s; pz[c][v]*=s; }
          }
        }

    // ---- Primal update per channel ----
    for (int c=0;c<C;++c){
      Eigen::VectorXf u_next = u[c];
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
      for (int z=0; z<NZ; ++z)
        for (int y=0; y<NY; ++y)
          for (int x=0; x<NX; ++x){
            const int v = idx(x,y,z);
            if (mask[v]==0.0f) continue;

            float dxm = px[c][v];
            float dym = py[c][v];
            float dzm = pz[c][v];
            if (x>0)   { int vm=idx(x-1,y,z); if (mask[vm]!=0.0f) dxm -= px[c][vm]; }
            if (y>0)   { int vm=idx(x,y-1,z); if (mask[vm]!=0.0f) dym -= py[c][vm]; }
            if (z>0)   { int vm=idx(x,y,z-1); if (mask[vm]!=0.0f) dzm -= pz[c][vm]; }
            const float divp = dxm+dym+dzm;

            u_next[v] = (u[c][v] + tau*divp + tau*mu*u0[c][v]) / (1.0f + tau*mu);
          }
      u_bar[c] = u_next + theta*(u_next - u[c]);
      u[c].swap(u_next);
    }
  }

  for (int c=0;c<C;++c)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int v=0; v<V; ++v)
      if (mask[v]!=0.0f) P(c,v) = u[c][v];
}

} // namespace qmri
