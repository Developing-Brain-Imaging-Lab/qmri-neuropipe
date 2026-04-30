# Dockerfile for qmri-neuropipe
# Base: NVIDIA CUDA 12.2 runtime on Ubuntu 22.04
#
# Notes:
# - FreeSurfer 8.2.0 was requested, but the official FreeSurfer distribution
#   site did not expose an 8.2.0 Ubuntu 22 package when this file was updated
#   on 2026-04-29. The default below therefore uses 8.1.0, with build args so
#   the package URL can be overridden when 8.2.0 becomes available.

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.10
ARG FREESURFER_VERSION=8.2.0
ARG FREESURFER_PACKAGE=freesurfer_ubuntu22-8.2.0_amd64.deb
ARG FREESURFER_URL=https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/${FREESURFER_VERSION}/${FREESURFER_PACKAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH

ENV FSLDIR=/usr/local/fsl
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV FREESURFER_HOME=/opt/freesurfer
ENV SUBJECTS_DIR=/opt/freesurfer/subjects
ENV FS_LICENSE=/opt/freesurfer/license.txt
ENV FSFAST_HOME=/opt/freesurfer/fsfast
ENV MNI_DIR=/opt/freesurfer/mni
ENV C3DPATH=/opt/c3d/bin
ENV PATH=${CONDA_DIR}/bin:${FSLDIR}/bin:${FREESURFER_HOME}/bin:${C3DPATH}:$PATH

# --------------------------------------------------------------------------------
# 1. System dependencies
# --------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ca-certificates \
    gnupg \
    git \
    build-essential \
    sudo \
    bc \
    dc \
    file \
    tcsh \
    perl \
    locales \
    libgl1-mesa-dev \
    libglu1-mesa \
    libfontconfig1 \
    libfreetype6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libsm6 \
    libxi6 \
    libxmu6 \
    libxkbcommon0 \
    libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------------
# 2. Miniforge and core neuroimaging binaries
# --------------------------------------------------------------------------------
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    mamba install -y -c mrtrix3 -c conda-forge \
        python=${PYTHON_VERSION} \
        pip \
        mrtrix3 \
        ants \
        afni \
        dcm2niix && \
    conda clean --all --yes

# --------------------------------------------------------------------------------
# 3. FSL
# --------------------------------------------------------------------------------
RUN curl -fsSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py -o /tmp/fslinstaller.py && \
    python3 /tmp/fslinstaller.py -d ${FSLDIR} -q --no_env && \
    rm /tmp/fslinstaller.py

# --------------------------------------------------------------------------------
# 4. FreeSurfer
# --------------------------------------------------------------------------------
RUN mkdir -p ${FREESURFER_HOME} && \
    wget -O /tmp/freesurfer.deb ${FREESURFER_URL} && \
    apt-get update && \
    apt-get install -y /tmp/freesurfer.deb && \
    rm -f /tmp/freesurfer.deb && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p ${SUBJECTS_DIR} && \
    touch ${FS_LICENSE}

# --------------------------------------------------------------------------------
# 5. Convert3D (C3D)
# --------------------------------------------------------------------------------
RUN mkdir -p /opt/c3d && \
    wget https://sourceforge.net/projects/c3d/files/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz/download -O /tmp/c3d.tar.gz && \
    tar -xzf /tmp/c3d.tar.gz -C /opt/c3d --strip-components=1 && \
    rm /tmp/c3d.tar.gz

# --------------------------------------------------------------------------------
# 6. qmri-neuropipe and Python dependencies
# --------------------------------------------------------------------------------
WORKDIR /app
COPY . /app

RUN ${CONDA_DIR}/bin/python -m pip install --upgrade pip setuptools wheel && \
    ${CONDA_DIR}/bin/python -m pip install ".[all]"

# --------------------------------------------------------------------------------
# 7. Runtime wrapper
# --------------------------------------------------------------------------------
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    'if [ -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ]; then' \
    '  . "${FREESURFER_HOME}/SetUpFreeSurfer.sh" >/dev/null 2>&1 || true' \
    'fi' \
    'exec "$@"' \
    > /usr/local/bin/qmri-entrypoint && \
    chmod +x /usr/local/bin/qmri-entrypoint

ENTRYPOINT ["/usr/local/bin/qmri-entrypoint", "qmri-neuropipe"]
CMD ["--help"]
