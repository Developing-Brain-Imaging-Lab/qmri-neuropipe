# Diffusion Model Fitting

`qmri-neuropipe` supports quantitative parameter mapping for various diffusion models. These models can be enabled in the pipeline configuration and run after standard preprocessing.

## Supported Models

Click on a model below for detailed configuration and usage instructions.

*   [**DTI (Diffusion Tensor Imaging)**](models/dti.md)
    *   Standard tensor fitting (FA, MD, AD, RD).
    *   Backends: `dipy`, `fsl`, `mrtrix`.

*   [**FWE-DTI (Free-Water Elimination DTI)**](models/fwe_dti.md)
    *   DTI corrected for partial volume effects of free water (CSF/Edema).
    *   Backend: `dipy`.

*   [**DKI (Diffusion Kurtosis Imaging)**](models/dki.md)
    *   Quantifies non-Gaussian diffusion using the kurtosis tensor.
    *   Backend: `dipy`.

*   [**NODDI (Neurite Orientation Dispersion and Density Imaging)**](models/noddi.md)
    *   Estimates neurite density and orientation dispersion.
    *   Features: Standard NODDI, **SMT-NODDI**, External FISO Constraints.
    *   Backend: `dmipy`, `amico`.

*   [**NEXI (Neurite Exchange Imaging)**](models/nexi.md)
    *   Estimates neurite exchange time and compartment diffusivities.
    *   Backend: `nexi`.

*   [**MAP-MRI (Mean Apparent Propagator MRI)**](models/mapmri.md)
    *   Reconstructs the full Ensemble Average Propagator (EAP).
    *   Backend: `dipy`.

*   [**SANDI (Soma and Neurite Density Imaging)**](models/sandi.md)
    *   Estimates soma (cell body) density, useful for gray matter.
    *   Backend: `amico`.

*   [**CSD (Constrained Spherical Deconvolution)**](models/csd.md)
    *   Estimates Fiber Orientation Distributions (FODs) to resolve crossing fibers.
    *   Backend: `mrtrix3`.

## Tractography & Segmentation

*   [**TractSeg**](models/tractseg.md)
    *   Deep learning-based bundle segmentation and tracking.
    *   Features: Initial MNI registration with inverse warping, bundle-specific tracking.
*   [**PyAFQ**](models/pyafq.md)
    *   Automated Fiber Quantification for infant and adult data.

## General Configuration

To enable any model, add a `modeling` section to your `preproc.yaml`:

```yaml
dmri:
  modeling:
    dti:
      enabled: true
      method: "dipy"
    noddi:
      enabled: true
      model_type: "smt"
```

## Modeling-Level Gradient Nonlinearity Map

When preprocessing outputs already exist (and are skipped), you can still
force modeling to use a gradient nonlinearity tensor map by enabling the
modeling-level GNL block. This will compute (or use) a tensor map and pass
it to DIPY/FSL backends during model fitting. When enabled (either in
preprocessing or modeling), the pipeline prefers a cached GNL tensor in the
preprocessed output directory and only recomputes if missing or forced.

```yaml
dmri:
  modeling:
    grad_nonlin:
      enabled: true
      coeff_file: /path/to/coeffs.dat
      force: false
      # Optional: use an existing map instead of computing it
      # map_path: /path/to/gnl_tensor.nii.gz
```
