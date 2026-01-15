#pragma once
#include <Eigen/Dense>
#include <random>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace qmri {

// -------------- cost = 0.5*||r||^2 + Gaussian prior --------------
template <typename Model>
inline double eval_cost_with_prior(const Model& vm,
                                   const Eigen::VectorXf& xobs,
                                   const Eigen::VectorXf& p,
                                   const SRCTag& tag)
{
  // VoxelModel is expected to define:
  //   - int num_params() const;
  //   - int num_obs() const;
  //   - using AutoDiffFunctor (or Numeric) with ctor (vm, xobs)
  //     and operator()(const double* p, double* r)
  const int m = vm.num_obs();
  std::vector<double> r(m);
  std::vector<double> pd(p.size());
  for (int i=0;i<p.size();++i) pd[i] = p[i];

  // Construct functor (assumes your functor accepts doubles)
  typename Model::AutoDiffFunctor F(vm, xobs);
  F(pd.data(), r.data());

  double data_term = 0.0;
  for (int i=0;i<m;++i) data_term += r[i]*r[i];
  data_term *= 0.5;

  double prior_term = 0.0;
  if (tag.prior_mean.size() == p.size() &&
      tag.prior_sigma.size() == p.size() &&
      tag.prior_weight > 0.0) {
    for (int j=0;j<p.size();++j) {
      const double z = (p[j] - tag.prior_mean[j]) / std::max(1e-12f, tag.prior_sigma[j]);
      prior_term += 0.5 * z*z;
    }
    prior_term *= tag.prior_weight;
  }
  return data_term + prior_term;
}

// ------------------------- helpers ----------------------------
inline Eigen::VectorXf clip_to_box(const Eigen::VectorXf& p,
                                   const Eigen::VectorXf& L,
                                   const Eigen::VectorXf& U)
{
  Eigen::VectorXf q = p;
  for (int i=0;i<p.size();++i) {
    if (std::isfinite(L[i])) q[i] = std::max(q[i], L[i]);
    if (std::isfinite(U[i])) q[i] = std::min(q[i], U[i]);
  }
  return q;
}

inline void enforce_positive(Eigen::VectorXf& p,
                             const Eigen::ArrayXi& mask,
                             float floor_val = 1e-6f)
{
  if (mask.size() != p.size()) return;
  for (int i=0;i<p.size();++i) if (mask[i]) p[i] = std::max(p[i], floor_val);
}

// ------------------------ main solver -------------------------
template <typename Model>
Eigen::VectorXf solve_voxel(const Model& vm,
                            const Eigen::VectorXf& xobs,
                            const Eigen::VectorXf& p0,
                            const SRCTag& tag_in)
{
  SRCTag tag = tag_in; // local copy (we will adjust)
  const int n = vm.num_params();
  Eigen::VectorXf best_p = p0;
  Eigen::VectorXf L(n), U(n);

  // Initialize [L,U]
  if (tag.use_bounds) {
    L = vm.lower_bounds(); U = vm.upper_bounds();
  } else {
    // default loose box around p0
    L = p0.array() * 0.1f;
    U = p0.array() * 10.0f;
    for (int i=0;i<n;++i) {
      if (!std::isfinite(L[i])) L[i] = -1e6f;
      if (!std::isfinite(U[i])) U[i] =  1e6f;
    }
  }

  // Positive mask → log-space sampling
  const bool use_log = (tag.positive_mask.size()==n) && (tag.positive_mask.any());

  // RNG setup (per-thread engines)
  std::vector<std::mt19937> engines;
  const int nthreads =
#ifdef _OPENMP
    std::max(1, omp_get_max_threads());
#else
    1;
#endif
  engines.reserve(nthreads);
  for (int t=0;t<nthreads;++t) engines.emplace_back(tag.seed + 1337u * t);

  auto sample_candidate = [&](std::mt19937& rng,
                              const Eigen::VectorXf& Lc,
                              const Eigen::VectorXf& Uc,
                              const Eigen::VectorXf& best,
                              Eigen::VectorXf& out) {
    std::uniform_real_distribution<float> U01(0.f, 1.f);
    out.resize(n);

    const bool use_prior = (U01(rng) < tag.prior_mix) &&
                           (tag.prior_mean.size()==n) &&
                           (tag.prior_sigma.size()==n);

    if (use_log && tag.positive_mask.size()==n) {
      // log-space boxes for positive params
      Eigen::VectorXf Llog = (Lc.array().max(1e-12f)).log();
      Eigen::VectorXf Ulog = (Uc.array().max(1e-12f)).log();
      for (int j=0;j<n;++j) {
        float z = U01(rng);
        float val;
        if (use_prior) {
          std::normal_distribution<float> N(tag.prior_mean[j]>0? std::log(std::max(1e-12f, tag.prior_mean[j])) : 0.f,
                                            tag.prior_sigma[j]>0? tag.prior_sigma[j] : 1.f);
          val = N(rng);
          // small jitter around current best (in log space)
          std::normal_distribution<float> J(0.f, tag.jitter_scale * std::max(1e-6f, Ulog[j]-Llog[j]));
          val += J(rng);
        } else {
          val = Llog[j] + z * (Ulog[j] - Llog[j]);
        }
        // clip to log-box and exponentiate
        val = std::min(std::max(val, Llog[j]), Ulog[j]);
        out[j] = std::exp(val);
      }
    } else {
      for (int j=0;j<n;++j) {
        float z = U01(rng);
        float val;
        if (use_prior) {
          std::normal_distribution<float> N(tag.prior_mean[j], std::max(1e-6f, tag.prior_sigma[j]));
          val = N(rng);
          std::normal_distribution<float> J(0.f, tag.jitter_scale * std::max(1e-6f, Uc[j]-Lc[j]));
          val += J(rng);
        } else {
          val = Lc[j] + z * (Uc[j] - Lc[j]);
        }
        out[j] = std::min(std::max(val, Lc[j]), Uc[j]);
      }
    }
    if (tag.use_bounds) out = clip_to_box(out, Lc, Uc);
    enforce_positive(out, tag.positive_mask);
  };

  // Evaluate initial best
  double best_cost = eval_cost_with_prior(vm, xobs, best_p, tag);

  int stall = 0;
  for (int it=0; it<tag.max_iters; ++it) {
    const int Np = tag.population;
    std::vector<Eigen::VectorXf> cand(Np, Eigen::VectorXf(n));
    std::vector<double>          cost(Np, std::numeric_limits<double>::infinity());

    // Contracted region around best: shrink [L,U] around current best by tag.contract_q
    Eigen::VectorXf Lc(n), Uc(n);
    for (int j=0;j<n;++j) {
      const float mid = best_p[j];
      const float half = 0.5f * (U[j] - L[j]) * tag.contract_q;
      Lc[j] = mid - half;
      Uc[j] = mid + half;
      if (tag.use_bounds) {
        Lc[j] = std::max(Lc[j], L[j]);
        Uc[j] = std::min(Uc[j], U[j]);
      }
    }

    // Sample + evaluate (parallel over candidates)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i=0; i<Np; ++i) {
#ifdef _OPENMP
      int tid = omp_get_thread_num();
#else
      int tid = 0;
#endif
      auto& rng = engines[tid];
      sample_candidate(rng, Lc, Uc, best_p, cand[i]);
      cost[i] = eval_cost_with_prior(vm, xobs, cand[i], tag);
    }

    // Find elites
    const int K = std::max(1, static_cast<int>(std::round(tag.elite_frac * Np)));
    std::vector<int> idx(Np); std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin()+K, idx.end(),
                     [&](int a, int b){ return cost[a] < cost[b]; });
    idx.resize(K);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b){ return cost[a] < cost[b]; });

    // Update best
    const double old_best = best_cost;
    if (cost[idx[0]] < best_cost) {
      best_cost = cost[idx[0]];
      best_p    = cand[idx[0]];
      stall = 0;
    } else {
      stall++;
    }

    // Update global [L,U] to elite envelope (slightly padded)
    Eigen::VectorXf Lnew = U;  // start inverted
    Eigen::VectorXf Unew = L;
    for (int j=0;j<n;++j) {
      for (int k=0;k<K;++k) {
        Lnew[j] = std::min(Lnew[j], cand[idx[k]][j]);
        Unew[j] = std::max(Unew[j], cand[idx[k]][j]);
      }
      const float pad = 0.05f * (Unew[j] - Lnew[j] + 1e-6f);
      Lnew[j] -= pad; Unew[j] += pad;
      if (tag.use_bounds) {
        Lnew[j] = std::max(Lnew[j], L[j]);
        Unew[j] = std::min(Unew[j], U[j]);
      }
    }
    L = Lnew; U = Unew;

    // Stopping
    const double rel = std::abs(old_best - best_cost) / (std::abs(old_best) + 1e-12);
    if (rel < tag.tol_rel) stall++;
    if (stall >= tag.max_stall) break;
  }

  return best_p;
}

} // namespace qmri
