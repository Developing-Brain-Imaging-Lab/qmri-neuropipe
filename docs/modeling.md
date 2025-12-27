# Diffusion Model Fitting

`qmri-neuropipe` supports quantitative parameter mapping for various diffusion models. These models can be enabled in the pipeline configuration and run after standard preprocessing.

## Available Models

### 1. DTI (Diffusion Tensor Imaging)
Estimates the diffusion tensor and derived metrics (FA, MD, AD, RD).

**Backends:**
- `dipy` (Default): Uses Weighted Linear Least Squares (WLLS) by default.
- `fsl`: Uses `dtifit`. (Requires FSL installed).
- `mrtrix`: Uses `dwi2tensor` and `tensor2metric`. (Requires MRtrix3 installed).

**Configuration:**
```yaml
dmri:
  modeling:
    dti:
      enabled: true
      method: "dipy"  # or "fsl", "mrtrix"
      parameters:
        sub_method: "WLLS" # For dipy: WLLS, OLS, NLLS, RESTORE
        metrics: ["fa", "md", "ad", "rd", "color_fa"]
```

### 2. DKI (Diffusion Kurtosis Imaging)
Estimates diffusion and kurtosis tensors.

**Backend:** `dipy`

**Configuration:**
```yaml
dmri:
  modeling:
    dki:
      enabled: true
      method: "dipy"
      parameters:
        metrics: ["mk", "ak", "rk", "fa", "md"]
```

### 3. NODDI (Neurite Orientation Dispersion and Density Imaging)
Estimates microstructural indices (ICVF, ODI, FISO).

**Backends:**
- `dmipy`: Python-based fitting.
- `amico`: Accelerated Microstructure Imaging via Convex Optimization. (Requires AMICO installed).

**Configuration:**
```yaml
dmri:
  modeling:
    noddi:
      enabled: true
      method: "dmipy" # or "amico"
```

### 4. SANDI (Soma and Neurite Density Imaging)
Estimates soma, neurite, and extra-cellular signal fractions.

**Backend:** `amico` (Requires AMICO installed).

**Configuration:**
```yaml
dmri:
  modeling:
    sandi:
      enabled: true
      method: "amico"
```

### 5. MAP-MRI (Mean Apparent Propagator MRI)
Estimates the Examplar Propagator and derived metrics (MSD, QIV, RTOP, RTAP, RTPP).

**Backend:** `dipy`

**Configuration:**
```yaml
dmri:
  modeling:
    mapmri:
      enabled: true
      method: "dipy"
      parameters:
        radial_order: 6
        laplacian_regularization: true
        laplacian_weighting: 0.2
```

## Output Structure

Model outputs are saved in the `dwi/` directory alongside preprocessed data, with filenames following BIDS convention:
- `sub-01_desc-dti_FA.nii.gz`
- `sub-01_desc-noddi_odi.nii.gz`
- `sub-01_desc-sandi_fsoma.nii.gz`
