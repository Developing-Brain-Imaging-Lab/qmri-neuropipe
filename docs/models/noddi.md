# NODDI (Neurite Orientation Dispersion and Density Imaging)

NODDI is a multi-compartment model that distinguishes between three signal environments:
1.  **Intra-cellular:** Restricted diffusion within neurites (modeled as sticks/cylinders).
2.  **Extra-cellular:** Hindered diffusion around neurites (modeled as Zeppelins).
3.  **Isotropic:** Free diffusion (CSF) (modeled as isotropic sphere).

## Backends

- **dmipy**: (Default) Flexible, Python-based fitting using the `Dmipy` library. Supports SMT-NODDI and parallel processing.
- **amico**: Accelerated Microstructure Imaging via Convex Optimization. Extremely fast but requires `amico` python package and setup.

## Configuration

```yaml
dmri:
  modeling:
    noddi:
      enabled: true
      method: "dmipy"
      # Global options
      model_type: "standard" # "standard" or "smt"
      distribution: "Watson" # "Watson" or "Bingham"
      parallel_diffusivity: 1.7e-9
      iso_diffusivity: 3.0e-9
      
      # Advanced
      fiso_map: null # Optional path/glob to external FISO constraint
```

### 1. Standard NODDI
The classic NODDI model using a Watson distribution for orientation dispersion.

```yaml
noddi:
  method: "dmipy"
  model_type: "standard"
  distribution: "Watson"
```

### 2. SMT-NODDI
Uses the Spherical Mean Technique to estimate microstructure parameters independent of fiber orientation (removing the need for a complex orientation distribution in the first step). This is often more robust for powder-averaged metrics.

```yaml
noddi:
  method: "dmipy"
  model_type: "smt"
```

### 3. External FISO Constraint
You can constrain the Free Water Volume Fraction (`fiso`) using an external map (e.g., derived from T2w segmentation or another model). This fixes the isotropic compartment per-voxel during fitting.

```yaml
noddi:
  fiso_map: "sub-{subject}_ses-{session}_desc-fiso.nii.gz" # Glob pattern supported
```

## Options

| Parameter | Desc | Default |
| :--- | :--- | :--- |
| `model_type` | `"standard"` or `"smt"` | `"standard"` |
| `distribution` | Orientation distribution function (`"Watson"` or `"Bingham"`) | `"Watson"` |
| `parallel_diffusivity` | Intrinsic diffusivity along neurites ($m^2/s$) | `1.7e-9` |
| `iso_diffusivity` | Diffusivity of free water ($m^2/s$) | `3.0e-9` |
| `solver` | Optimizer strategy (e.g. `"brute2fine"`, `"mix"`) | `"brute2fine"` |

## Outputs

- **ICVF (Intra-Cellular Volume Fraction):** `*_model-NODDI_icvf.nii.gz`
- **ODI (Orientation Dispersion Index):** `*_model-NODDI_odi.nii.gz`
- **ISO (Isotropic Fraction):** `*_model-NODDI_iso.nii.gz`
