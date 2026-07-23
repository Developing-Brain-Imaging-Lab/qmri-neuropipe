# Third-party notices

qmri-neuropipe depends on third-party software distributed under its own
license terms. This notice is informational and does not replace the license
files shipped by those projects.

## dmipy-fit

- Project: https://github.com/dmrai-lab/dmipy-fit
- Package constraint: `dmipy-fit>=2.1,<2.2`
- License: `AGPL-3.0-only OR LicenseRef-Commercial`

The open-source integration in qmri-neuropipe uses dmipy-fit under
AGPL-3.0-only.

## dmipy-sim

- Project: https://github.com/dmrai-lab/dmipy-sim
- License: `AGPL-3.0-only OR LicenseRef-Commercial`

dmipy-fit 2.1 declares dmipy-sim as a runtime dependency. qmri-neuropipe's
Phase 1 integration uses the analytical fitting engine; direct Monte Carlo
simulation support is not yet exposed.

## Corresponding source

Source releases and container images must identify the exact qmri-neuropipe
release tag or Git commit from which they were built. The source repository is:

https://github.com/Developing-Brain-Imaging-Lab/qmri-neuropipe
