#include <iostream>
#include <filesystem>
#include <vector>
#include <fstream>
#include <chrono>
#include <exception>
#include <numeric>

#include <omp.h>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

#include <docopt/docopt.h>

#include "qmri/models/despot1_hifi.hpp"
#include "qmri/exec/batch_executor.hpp"
#include "qmri/exec/batch_executor_progress.hpp"
#include "qmri/exec/batch_executor_masked.hpp"

#include "qmri/core/optimizer_locking.hpp"
#include "qmri/core/optimizer_bridge.hpp"
#include "qmri/core/optimizer_tags.hpp"

#include "qmri/exec/data_fit.hpp"
#include "qmri/util/progress.hpp"


#include "qmri/regularizers/admm_laplacian.hpp"
#include "qmri/regularizers/tv_rof.hpp"
#include "qmri/io/itk_nifti.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;

static const char USAGE[] = R"(qmri_fit_despot1_hifi

Usage:
  qmri_fit_despot1 --spgr=<file> --irspgr=<file> --params=<json> --b1=<file> --out_dir=<dir>
                   [--algo=<a>] [--out_base=<file>] [--mask=<file>] [--lambda=<l>] [--mu=<m>] [--reg=<r>] 
                   [--tv-mode=<tm>] [--tv-penalty=<tp>] [--huber-eps=<e>] [--tau=<t>] [--sigma=<s>] [--theta=<th>] 
                   [--outer=<k>] [--tv-iters=<n>] [--chunk=<n>] [--log-json=<file>] [--b1-percent] [--nthreads=<cpu>] [--progress] [--verbose]
  qmri_fit_despot1 (-h | --help)

Options:
  -h --help          Show this screen.
  --spgr=<file>      SPGR Data file
  --irspgr=<file>    IRSPGR Data file
  --params=<json>    Acquisition Parameters within JSON file
  --out_dir=<file>   Output directory
  --b1=<file>        External B1 map NIfTI (float).
  --mask=<file>      Mask NIfTI; fit where mask==1.
  --out_base=<file>  Output basename [default: DESPOT1_]
  --algo=<a>         CERES | NLOPT | OLS | WLS [default: CERES].
  --nthreads=<cpu>   Number of Cores [default: 2].
  --b1-percent       Treat B1 map as percent (100 -> 1.0) [default: false].
  --lambda=<l>       TV weight [default: 0.0].
  --mu=<m>           Data fidelity (ROF) weight, only for TV [default: 1.0].
  --reg=<r>          none | lap | tv [default: none].
  --tv-mode=<tm>     scalar | vectorial [default: scalar].
  --tv-penalty=<tp>  iso | huber [default: iso].
  --huber-eps=<e>    Huber epsilon (smoothing) [default: 0.0].
  --tau=<t>          TV primal step size [default: 0.02].
  --sigma=<s>        TV dual step size [default: 0.02].
  --theta=<th>       TV over-relaxation [default: 1.0].
  --outer=<k>        Alternating outer iterations (data + prior) [default: 1].
  --tv-iters=<n>     TV iterations per outer step [default: 30].
  --chunk=<n>        Chunk size for voxel progress [default: 100].
  --log-json=<file>  Write JSON summary (config + per-outer records) [default: ].
  --progress         Show stage progress bars [default: false].
  --verbose          Show output [default: false].
  

  
)";

//static std::pair<double,double> compute_rmse_stats(const Eigen::MatrixXf& P,
//                                                   const Eigen::MatrixXf& X,
//                                                   const qmri::DESPOT1HiFi& model){
//  int V = X.cols();
//  Eigen::VectorXd rn(V);
//  #pragma omp parallel for schedule(static)
//  for (int v=0; v<V; ++v){
//    Eigen::VectorXf p = P.col(v);
//    Eigen::VectorXf r = model.residuals(p, X.col(v));
//    rn[v] = std::sqrt(r.template cast<double>().squaredNorm() / std::max(1, (int)r.size()));
//  }
//  double mean = rn.mean();
//  double var = 0.0;
//  for (int v=0; v<V; ++v){ double d = rn[v] - mean; var += d*d; }
//  var /= std::max(1, V-1);
//  return {mean, std::sqrt(var)};
//}

int main(int argc, char** argv){

    std::vector<std::string> argvec;
    for (int i=1;i<argc;++i) argvec.emplace_back(argv[i]);
    if (argvec.empty()) argvec.emplace_back("--help");
  
    auto args = docopt::docopt(USAGE, argvec, true, "qmri_fit_despot1 0.6");
        
    const fs::path    spgr_path = args["--spgr"].asString();
    const fs::path    irspgr_path = args["--irspgr"].asString();
    const fs::path    par_path = args["--params"].asString();
    const fs::path    out_dir  = args["--out_dir"].asString();
    
    const std::string out_base   = args["--out_base"].asString();
    const std::string algo       = args["--algo"].asString();
    const std::string mask_path  = args["--mask"].asString();
    const std::string b1_path    = args["--b1"].asString();
    const bool        b1_percent = args["--b1-percent"].asBool();
    
    const bool        use_progress   = args["--progress"].asBool();
    const int         chunk          = (int)args["--chunk"].asLong();
    const int         ncpu           = (int)args["--nthreads"].asLong();
    const bool        verbose        = args["--verbose"].asBool();
    
    const std::string log_json   = args["--log-json"].asString();

    //Regularization options
    const std::string reg        = args["--reg"].asString();
    const std::string tv_mode    = args["--tv-mode"].asString();
    const std::string tv_penalty = args["--tv-penalty"].asString();
    const float       lambda     = std::stof(args["--lambda"].asString());
    const float       mu         = std::stof(args["--mu"].asString());
    const float       huber_eps  = std::stof(args["--huber-eps"].asString());
    const float       tv_tau     = std::stof(args["--tau"].asString());
    const float       tv_sigma   = std::stof(args["--sigma"].asString());
    const float       tv_theta   = std::stof(args["--theta"].asString());
    const int         outer_iters= (int)args["--outer"].asLong();
    const int         tv_iters   = (int)args["--tv-iters"].asLong();
    

    //Create output directory if it doesn't exist
    omp_set_num_threads(ncpu);
    fs::create_directories(out_dir);
    std::unique_ptr<qmri::ProgressBar> progress;
    
    //Load in the Acquisition Parameters
    json acq_params = json::parse(std::ifstream(par_path.string()));
    std::vector<float> spgrFlips, spgrTRs, irspgrFlips, irspgrTRs, irspgrTIs, irspgrETL;
    for(auto &array : acq_params["SPGR"]) {
        auto fa = array["FlipAngle"].get<std::vector<double>>();
        auto tr = array["RepetitionTime"].get<std::vector<double>>();
        spgrFlips.insert(spgrFlips.end(), fa.begin(), fa.end());
        spgrTRs.insert(spgrTRs.end(), tr.begin(), tr.end());
    }
    for(auto &array : acq_params["IRSPGR"]) {
        auto fa = array["FlipAngle"].get<std::vector<double>>();
        auto tr = array["RepetitionTime"].get<std::vector<double>>();
        auto ti = array["InversionTime"].get<std::vector<double>>();
        auto etl = array["EchoTrainLength"].get<std::vector<double>>();
        
        irspgrFlips.insert(irspgrFlips.end(), fa.begin(), fa.end());
        irspgrTRs.insert(irspgrTRs.end(), tr.begin(), tr.end());
        irspgrTIs.insert(irspgrTIs.end(), ti.begin(), ti.end());
        irspgrETL.insert(irspgrETL.end(), etl.begin(), etl.end());
    }
    
    //Number of observations
    const int Nspgr = static_cast<int>(spgrFlips.size());
    const int Nirspgr = static_cast<int>(irspgrTIs.size());
    const int Nobs = static_cast<int>(Nspgr + Nirspgr);
    
    //Read in the SPGR data
    using Img4D = qmri::io::Image<float,4>;
    auto reader = qmri::io::Reader<float,4>::New();
    reader->SetFileName(spgr_path.string());
    reader->Update();
    auto spgr4d = reader->GetOutput();

    auto region = spgr4d->GetLargestPossibleRegion();
    auto size   = region.GetSize();
    int NX = static_cast<int>(size[0]);
    int NY = static_cast<int>(size[1]);
    int NZ = static_cast<int>(size[2]);
    int NT = static_cast<int>(size[3]);
    int V  = NX * NY * NZ; // Number of Voxels;

    if (NT != Nspgr) { std::cerr << "Mismatch between the number of SPGR Input Images and Nobs from parameters\n"; return 3; }

    Eigen::MatrixXf X(Nobs,V), X_spgr(Nspgr,V), X_irspgr(Nirspgr,V);
    
    qmri::io::pack_soa(spgr4d, X_spgr);
    
    //Read in the IRSPGR data
    auto reader2 = qmri::io::Reader<float,4>::New();
    reader2->SetFileName(irspgr_path.string());
    reader2->Update();
    auto irspgr4d = reader2->GetOutput();
    qmri::io::pack_soa(irspgr4d, X_irspgr);
    
    X << X_spgr, X_irspgr;
    
    
    //Read in the Mask and determine voxels that will be solved.
    std::vector<int> active_vox; active_vox.reserve(V);
    Eigen::VectorXi maskV = Eigen::VectorXi::Zero(V);
    if (!mask_path.empty()) {
        using Img3D = qmri::io::Image<float,3>;
        auto mreader = qmri::io::Reader<float,3>::New();
        mreader->SetFileName(mask_path);
        mreader->Update();
        auto mimg = mreader->GetOutput();
        auto mreg = mimg->GetLargestPossibleRegion();
        auto msz  = mreg.GetSize();
        if ((int)msz[0]!=NX || (int)msz[1]!=NY || (int)msz[2]!=NZ) { std::cerr << "Mask dims do not match image dims\n"; return 4; }
        
        int idx=0;
        itk::ImageRegionConstIterator<Img3D> it(mimg, mreg);
        for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++idx) {
            int on = (it.Get() > 0.5f) ? 1 : 0;
            maskV[idx] = on;
            if (on) active_vox.push_back(idx);
        }
    }else{
        active_vox.resize(V);
        std::iota(active_vox.begin(), active_vox.end(), 0);
        maskV.setOnes();
    }
    
    int Vsel = (int)active_vox.size();
    if (Vsel == 0) { std::cerr << "Mask selected 0 voxels.\n"; return 5; }

    Eigen::MatrixXf P = Eigen::MatrixXf::Zero(3, V);
    Eigen::MatrixXf F = Eigen::MatrixXf::Zero(1, V);

    std::vector<json> records;
    qmri::HiFiParams params(Eigen::Map<const Eigen::VectorXf>(spgrFlips.data(), Nspgr),
                            Eigen::Map<const Eigen::VectorXf>(spgrTRs.data(), Nspgr),
                            Eigen::Map<const Eigen::VectorXf>(irspgrTRs.data(), Nirspgr),
                            Eigen::Map<const Eigen::VectorXf>(irspgrTIs.data(), Nirspgr),
                            Eigen::Map<const Eigen::VectorXf>(irspgrFlips.data(), Nirspgr),
                            Eigen::Map<const Eigen::VectorXf>(irspgrETL.data(), Nirspgr));
                                   
    qmri::HiFiBinder hifi(params);
    qmri::AnyTag tag = qmri::make_tag_from_cli(args);
    
    //Print some information
    if(verbose){
        
        int id = 4;
        std::string indent;
        for(int i=0; i<id; i++){ indent += " "; }

        params.print(std::cout, id);
        
        std::cout << indent << "Input Images: " << std::endl;
        std::cout << indent << indent << "SPGR Images: " << std::filesystem::absolute(spgr_path) << std::endl << std::endl;
        std::cout << indent << indent << "IR-SPGR Images: " << std::filesystem::absolute(irspgr_path) << std::endl << std::endl;
        
        if(!mask_path.empty())
            std::cout << indent << "Mask Image: " << std::filesystem::absolute(mask_path) << std::endl << std::endl;

        std::cout << indent << "Number of Threads: " << ncpu << std::endl;
        std::cout << indent << "Fitting Algorithm: " << algo << std::endl;
        std::cout << std::endl;
        
    }
        
    auto t0 = std::chrono::steady_clock::now();
    if (use_progress) {
        qmri::ProgressBar pb(static_cast<int>(active_vox.size()), "Fitting: ", 40, 4);
        
        std::visit([&](const auto& concrete_tag){
            qmri::fit_data(hifi, X, P, active_vox, chunk, concrete_tag,
                       [&](int done, int){ pb.update(done); });
        }, tag);
        
        pb.finish();
        
    } else {
      std::visit([&](const auto& concrete_tag){
        qmri::fit_data(hifi, X, P, active_vox, chunk, concrete_tag);
      }, tag);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    if(verbose){
        
        int id = 4;
        std::string indent;
        for(int i=0; i<id; i++){ indent += " "; }
        
        std::cout << indent << "Finished in: " << ms << " milliseconds" << std::endl;
        std::cout << std::endl;
        
    }
    
//    auto [rm, rs] = compute_rmse_stats(P, X, model);
//    records.push_back(json{{"stage","data_fit"},{"outer",-1},{"elapsed_ms",ms},{"rmse_mean",rm},{"rmse_std",rs}});
//      

  for (int outer=0; outer<outer_iters; ++outer) {
      int current_outer = outer+1;
      
      //if (progress) { std::cout << "=== Outer " << current_outer << "/" << outer_iters << " ===" << std::endl; print_bar("Data fit", 0, 1); }
      std::visit([&](const auto& concrete_tag){
        qmri::fit_data(hifi, X, P, active_vox, chunk, concrete_tag);
      }, tag);

      
      //if (progress) print_bar("Data fit", 1, 1);
      
      if (lambda > 0.f && (reg!="none" && reg!="NONE")) {
          
          auto t0_reg = std::chrono::steady_clock::now();
          
          // Mask as float for Eigen::Ref binding
          Eigen::VectorXf maskF = maskV.cast<float>();
          
          if (reg=="lap" || reg=="LAP") {
              if (use_progress) {
                  qmri::ProgressBar pbr(1, "Laplacian", 40, 4);
                  pbr.update(0);
                  qmri::admm_laplacian_regularize_masked(P, NX, NY, NZ, lambda, maskF, /*iters=*/10);
                  pbr.update(1);
                  pbr.finish();
              } else {
                  qmri::admm_laplacian_regularize_masked(P, NX, NY, NZ, lambda, maskF, /*iters=*/10);
              }
          } else if (reg=="tv" || reg=="TV") {
              Eigen::MatrixXf P0 = P;
              const bool use_huber = (tv_penalty=="huber" || tv_penalty=="HUBER");
              const std::string tv_label = (tv_mode=="vectorial" || tv_mode=="VECTORIAL")
              ? "TV (vectorial)" : "TV (scalar)";
              
              if (use_progress) {
                  qmri::ProgressBar pbtv(tv_iters, tv_label, 40, 4);
                  for (int t=0; t<tv_iters; ++t) {
                      if (tv_mode=="vectorial" || tv_mode=="VECTORIAL") {
                          qmri::tv_rof_vectorial_cp_masked(P, P0, NX, NY, NZ, lambda, mu, maskF,
                                                           /*iters=*/1, tv_tau, tv_sigma, tv_theta,
                                                           use_huber ? huber_eps : 0.f);
                      } else {
                          qmri::tv_rof_scalar_cp_masked(P, P0, NX, NY, NZ, lambda, mu, maskF,
                                                        /*iters=*/1, tv_tau, tv_sigma, tv_theta,
                                                        use_huber ? huber_eps : 0.f);
                      }
                      pbtv.update(t+1);
                  }
                  pbtv.finish();
              } else {
                  if (tv_mode=="vectorial" || tv_mode=="VECTORIAL") {
                      qmri::tv_rof_vectorial_cp_masked(P, P0, NX, NY, NZ, lambda, mu, maskF,
                                                       /*iters=*/tv_iters, tv_tau, tv_sigma, tv_theta,
                                                       use_huber ? huber_eps : 0.f);
                  } else {
                      qmri::tv_rof_scalar_cp_masked(P, P0, NX, NY, NZ, lambda, mu, maskF,
                                                    /*iters=*/tv_iters, tv_tau, tv_sigma, tv_theta,
                                                    use_huber ? huber_eps : 0.f);
                  }
              }
          }
          
          auto t1_reg = std::chrono::steady_clock::now();
          double ms_reg = std::chrono::duration<double, std::milli>(t1_reg - t0_reg).count();
          if (verbose) { std::cout << "Regularization done in " << ms_reg << " ms\n"; }
      }
    
    std::cout << "Outer iteration " << current_outer << " / " << outer_iters << " complete." << std::endl;
  }

    
    std::vector<std::string> output_paths(3);
    output_paths[0] = (out_dir/(out_base+"M0.nii.gz")).string();
    output_paths[1] = (out_dir/(out_base+"T1.nii.gz")).string();
    output_paths[2] = (out_dir/(out_base+"B1.nii.gz")).string();
    
    using Img3D = qmri::io::Image<float,3>;
    for(int i=0; i<3; i++)
        qmri::io::write_map<Img3D>(P.row(i).transpose(), NX,NY,NZ, output_paths[i]);
    
    if(verbose){
        int id = 4;
        std::string indent;
        for(int i=0; i<id; i++){ indent += " "; }
        
        std::cout << indent << "Writing Outputs:" << std::endl;
        std::cout << indent << indent << "M0 Map: " << std::filesystem::absolute(output_paths[0]) << std::endl;
        std::cout << indent << indent << "T1 Map: " << std::filesystem::absolute(output_paths[1]) << std::endl;
        std::cout << indent << indent << "B1 Map: " << std::filesystem::absolute(output_paths[2]) << std::endl;
        std::cout << std::endl;
        
    }
    
 
//
//  if (!log_json.empty()) {
//    json cfg = {
//      {"algo",algo},{"lambda",lambda},{"mu",mu},
//      {"reg",reg},{"tv_mode",tv_mode},{"tv_penalty",tv_penalty},
//      {"tau",tv_tau},{"sigma",tv_sigma},{"theta",tv_theta},
//      {"outer",outer_iters},{"tv_iters",tv_iters},
//      {"voxel_progress",voxel_progress},{"chunk",chunk},
//      {"voxels",V},{"observations",Nobs},
//      {"mask",!mask_path.empty()},{"b1_locked",lock_b1}
//    };
//    json out = {{"config",cfg},{"records",records}};
//    std::ofstream os(log_json);
//    os << out.dump(2) << std::endl;
//  }

  return 0;
}
