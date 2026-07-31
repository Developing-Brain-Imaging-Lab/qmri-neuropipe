# NEXI (Neurite Exchange Imaging)

For every accepted field and default, see {ref}`Diffusion model fitting <diffusion-model-fitting>`.

NEXI estimates neurite exchange time and diffusivities using the `nexi` Python package.

## Requirements

- `nexi` Python package (`pip install nexi`)
- A diffusion time file (`td_file`) in milliseconds (one value per volume)
- A low-b noise map (`lowb_noisemap`) computed from low-b data (b < 2 ms/um^2)

The NEXI implementation internally converts b-values > 500 to ms/um^2, so standard
b-values in s/mm^2 are supported without additional scaling.

## Configuration

```yaml
dmri:
  modeling:
    nexi:
      enabled: true
      method: nexi
      td_file: /path/to/td.txt
      lowb_noisemap: /path/to/lowb_noisemap.nii.gz
      metrics: [t_ex, di, de, f, sigma]
```

## Outputs

Outputs are saved in the model directory with BIDS-style suffixes:

- `TEX` (t_ex) — exchange time
- `DI` (di) — intra-neurite diffusivity
- `DE` (de) — extra-neurite diffusivity
- `F` (f) — intra-neurite volume fraction
- `SIGMA` (sigma) — noise parameter
