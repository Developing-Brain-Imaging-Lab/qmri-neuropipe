#pragma once
#include <string>
#include <docopt/docopt.h>

namespace qmri {

struct CeresTag { 
    int max_iterations = 100; 
    double ftol = 1e-8, gtol = 1e-8, xtol = 0.0; 
    bool verbose = false; 

};

struct NLoptTag { 
    std::string algorithm = "LD_LBFGS"; 
    int maxeval = 500; 
    double xtol_rel = 1e-6, ftol_rel = 0.0, maxtime = 0.0; 
    bool verbose = false; 

};

struct LinearTag { 
    bool weighted = false; 
};

// ------------------- SRC configuration tag -------------------
struct SRCTag {
  // Population / loop
  int    population      = 5000;     // candidates per iteration
  double elite_frac      = 0.2;     // top K = elite_frac * population
  int    max_iters       = 5;
  int    max_stall       = 5;       // stop if best not improving
  double contract_q      = 0.6;     // shrink multiplier on [L,U]
  double tol_rel         = 1e-4;    // relative improvement tol

  // Mixture sampling between uniform box and Gaussian prior
  double prior_mix       = 0.50;    // prob. to sample from Gaussian prior
  double jitter_scale    = 0.10;    // local Gaussian jitter around current best (% of (U-L))

  // Priors / bounds (size = num_params or 0 to ignore)
  Eigen::VectorXf lower;            // uniform bounds
  Eigen::VectorXf upper;            // uniform bounds
  bool use_bounds      = true;   // if true, clip to [lower,upper]

  Eigen::VectorXf prior_mean;       // Gaussian prior mean
  Eigen::VectorXf prior_sigma;      // Gaussian prior std (diag)
  double prior_weight    = 1.0;     // weight of Gaussian prior in cost (L2 penalty)

  // Parameter domain behavior
  Eigen::ArrayXi positive_mask;     // 1 => sample in log-space & enforce >0
  unsigned int   seed       = 42;
  bool   verbose            = false;
};

using AnyTag = std::variant<qmri::CeresTag, qmri::NLoptTag, qmri::LinearTag, qmri::SRCTag>;

AnyTag make_tag_from_cli(docopt::Options& args) {
  const std::string algo = args["--algo"].asString();
  if (algo == "CERES") {
      qmri::CeresTag t;
//      t.max_iterations = args["--ceres-max-iter"].asInt();
//      t.xtol           = args["--ceres-xtol"].asDouble();
//      t.ftol           = args["--ceres-ftol"].asDouble();
//      t.gtol           = args["--ceres-gtol"].asDouble();
//      t.verbose        = args["--verbose"].asBool();
      
      return t;
  } else if (algo == "NLOPT") {
      qmri::NLoptTag t;
      t.algorithm = args["--nlopt-alg"].asString();  // "LN_BOBYQA", etc.
//      t.max_eval  = args["--nlopt-max-eval"].asInt();
//      t.xtol_rel  = args["--nlopt-xtol-rel"].asDouble();
//      t.ftol_rel  = args["--nlopt-ftol-rel"].asDouble();
//      t.verbose   = args["--verbose"].asBool();
      
      return t;
  } else if (algo == "OLS" || algo == "WLS") {
      qmri::LinearTag t;
      if(algo == "WLS"){
          t.weighted = true;
      }else
          t.weighted = false;
      
    return t;
  } else if (algo == "SRC"){
      
      qmri::SRCTag t;
      
      return t;
      
  }
  throw std::runtime_error("Unknown --algo");
}

} // namespace qmri
