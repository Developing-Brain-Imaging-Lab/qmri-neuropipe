# PyAFQ (Automated Fiber Quantification)

PyAFQ is an automated structural connectivity and tractography analysis pipeline. It is particularly well-suited for both infant and adult datasets.

The current integration asks PyAFQ to generate and segment its own tractography.
It does not import the MRtrix whole-brain tractogram. Use the MRtrix
`tract_specific` workflow when MRtrix tractometry or SIFT2-weighted connectomes
are required.

## Configuration

```yaml
dmri:
  modeling:
    tractography:
      pyafq:
        enabled: true
        options:
          profile: "default"
          n_threads: 8
```

## Features

- **Tractography**: Automated generation of streamlines.
- **Tractometry**: Extraction of diffusion metrics (FA, MD, etc.) along the length of each tract.
- **BIDS Integration**: Fully compliant with BIDS derivatives standards.

## Requirements

PyAFQ requires `pyafq` to be installed. You can install it via:
```bash
pip install "qmri-neuropipe[pyafq]"
```
