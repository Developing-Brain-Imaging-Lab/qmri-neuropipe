# QMRI NextGen

From-scratch, modular qMRI library featuring:
- DESPOT1 model with analytic Jacobian
- Ceres (default), optional NLopt, and fast linear estimators
- OpenMP SoA executor
- ITK IO at the edges
- TV (Chambolle–Pock) and Laplacian (ADMM-like) regularizers
- Alternating loop: per-voxel data fit ↔ spatial prior

Build:
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j

Run:
  ./build/qmri_fit_despot1 spgr_4d.nii.gz params.json out --algo=CERES --lambda=0.05 --reg=tv --outer=3 --tv-iters=30

Note: replace external/docopt/docopt.h/.cpp with the official files for full CLI behavior.

- TV solver now supports ROF data term and vectorial TV via Chambolle–Pock. Use --mu and --tv-mode.

- TV CLI knobs: --tau, --sigma, --theta, --tv-penalty=iso|huber, --huber-eps; progress bar via --progress.

- Per-voxel progress: `--voxel-progress` (chunked OpenMP, default 8192). CSV logging via `--log-csv=path.csv`.
