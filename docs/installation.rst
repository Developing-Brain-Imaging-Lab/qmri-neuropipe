------------
Installation
------------
There are two ways to install *qmri-neuropipe*:

* within a `Manually Prepared Python Environment`_, or
* using container (Docker or Singularity);



Manually Prepared Python Environment
============================================

Make sure all of *qmri-neuropipe*'s `External Dependencies`_ are installed.
These tools must be installed and their binaries available in the
system's ``$PATH``.

External Dependencies
---------------------
*qmri-neuropipe* is written using Python 3.8 (or above)

*qmri-neuropipe* requires other neuroimaging software tools that are
not handled by the Python's packaging system (Pypi) used to deploy
the ``qmri-neuropipe`` package:

- FSL (version 6.0.7.7)
- ANTs (version 2.5.1)
- AFNI (version 24.0.05)
- `C3D <https://sourceforge.net/projects/c3d/>` (version 1.4.0)
- FreeSurfer (version 7.3.2, optional)
- Tortoise (version 4)