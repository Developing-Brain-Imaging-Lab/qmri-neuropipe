# Dockerfile for qmri-neuropipe
# Base: NVIDIA CUDA 12.2 runtime on Ubuntu 22.04
#
# Notes:
# - The default FreeSurfer package below targets version 8.2.0 for Ubuntu 22.

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ARG PYTHON_VERSION=3.10
ARG FREESURFER_VERSION=8.2.0
ARG FREESURFER_PACKAGE=freesurfer_ubuntu22-8.2.0_amd64.deb
ARG FREESURFER_URL=https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/${FREESURFER_VERSION}/${FREESURFER_PACKAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:$PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV FSLDIR=/usr/local/fsl
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV FREESURFER_HOME=/usr/local/freesurfer/${FREESURFER_VERSION}
ENV SUBJECTS_DIR=${FREESURFER_HOME}/subjects
ENV FS_LICENSE=${FREESURFER_HOME}/license.txt
ENV FSFAST_HOME=${FREESURFER_HOME}/fsfast
ENV MNI_DIR=${FREESURFER_HOME}/mni
ENV C3DPATH=/opt/c3d/bin
ENV TORTOISE_HOME=/opt/tortoise
ENV QMRI_FIT_HOME=/opt/qmri-fit
ENV DIPY_HOME=/opt/dipy
ENV LD_LIBRARY_PATH=${TORTOISE_HOME}/lib:${QMRI_FIT_HOME}/lib:$LD_LIBRARY_PATH
ENV PATH=${CONDA_DIR}/bin:${FSLDIR}/bin:${FREESURFER_HOME}/bin:${FREESURFER_HOME}/python/bin:${FREESURFER_HOME}/python/scripts:${C3DPATH}:${TORTOISE_HOME}/bin:${QMRI_FIT_HOME}/bin:$PATH

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
        numpy \
        cython \
        imagecodecs \
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
    FS_ACTUAL_HOME="" && \
    for candidate in \
        "${FREESURFER_HOME}" \
        "/usr/local/freesurfer/${FREESURFER_VERSION%.*}" \
        "/usr/local/freesurfer" \
        "/opt/freesurfer"; do \
        if [ -x "${candidate}/bin/mri_convert" ]; then FS_ACTUAL_HOME="${candidate}"; break; fi; \
    done && \
    if [ -z "${FS_ACTUAL_HOME}" ]; then echo "Could not locate FreeSurfer install root after package install." >&2; find /usr/local -maxdepth 3 -type f -name mri_convert >&2 || true; exit 1; fi && \
    mkdir -p ${SUBJECTS_DIR} && \
    touch ${FS_LICENSE} && \
    mkdir -p "${FS_ACTUAL_HOME}/models" && \
    wget -O "${FS_ACTUAL_HOME}/models/SuperSynth_August_2025.pth" "https://ftp.nmr.mgh.harvard.edu/pub/dist/lcnpublic/dist/SuperSynth_Iglesias_2025/SuperSynth_August_2025.pth" && \
    if [ -x "${FS_ACTUAL_HOME}/python/scripts/mri_synthseg" ] && [ ! -e "${FS_ACTUAL_HOME}/bin/mri_synthseg" ]; then ln -s "${FS_ACTUAL_HOME}/python/scripts/mri_synthseg" "${FS_ACTUAL_HOME}/bin/mri_synthseg"; fi

# --------------------------------------------------------------------------------
# 5. Convert3D (C3D)
# --------------------------------------------------------------------------------
RUN mkdir -p /opt/c3d && \
    wget https://sourceforge.net/projects/c3d/files/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz/download -O /tmp/c3d.tar.gz && \
    tar -xzf /tmp/c3d.tar.gz -C /opt/c3d --strip-components=1 && \
    rm /tmp/c3d.tar.gz && \
    mkdir -p /data /output /code

# --------------------------------------------------------------------------------
# 6. Optional TORTOISE binaries
# --------------------------------------------------------------------------------
COPY container-assets /opt/container-assets
RUN if [ -d /opt/container-assets/tortoise/bin ] && [ -f /opt/container-assets/tortoise/bin/CreateGradientNonlinearityBMatrix ]; then \
        mkdir -p ${TORTOISE_HOME} && \
        cp -a /opt/container-assets/tortoise/. ${TORTOISE_HOME}/ && \
        chmod -R a+rX ${TORTOISE_HOME} && \
        chmod a+rx ${TORTOISE_HOME}/bin/* || true && \
        echo "Installed TORTOISE binaries from /opt/container-assets/tortoise to ${TORTOISE_HOME}" && \
        ${TORTOISE_HOME}/bin/CreateGradientNonlinearityBMatrix --help >/dev/null 2>&1 || true; \
    else \
        echo "No local TORTOISE binaries found under /opt/container-assets/tortoise/bin; skipping TORTOISE install."; \
    fi

# --------------------------------------------------------------------------------
# 7. Optional qMRI fitting binary from qmri_nextgen
# --------------------------------------------------------------------------------
RUN QMRI_FIT_SRC=/opt/container-assets/qmri-fit; \
    QMRI_FIT_BIN=""; \
    if [ -f "${QMRI_FIT_SRC}/bin/qmri-fit" ]; then QMRI_FIT_BIN="${QMRI_FIT_SRC}/bin/qmri-fit"; \
    elif [ -f "${QMRI_FIT_SRC}/qmri-fit" ]; then QMRI_FIT_BIN="${QMRI_FIT_SRC}/qmri-fit"; \
    elif [ -f "${QMRI_FIT_SRC}/bin/qmri_fit" ]; then QMRI_FIT_BIN="${QMRI_FIT_SRC}/bin/qmri_fit"; \
    elif [ -f "${QMRI_FIT_SRC}/qmri_fit" ]; then QMRI_FIT_BIN="${QMRI_FIT_SRC}/qmri_fit"; fi; \
    if [ -n "${QMRI_FIT_BIN}" ]; then \
        mkdir -p ${QMRI_FIT_HOME} && \
        cp -a "${QMRI_FIT_SRC}/." ${QMRI_FIT_HOME}/ && \
        mkdir -p ${QMRI_FIT_HOME}/bin && \
        if [ ! -f ${QMRI_FIT_HOME}/bin/qmri-fit ] && [ -f ${QMRI_FIT_HOME}/qmri-fit ]; then mv ${QMRI_FIT_HOME}/qmri-fit ${QMRI_FIT_HOME}/bin/qmri-fit; fi && \
        if [ ! -f ${QMRI_FIT_HOME}/bin/qmri_fit ] && [ -f ${QMRI_FIT_HOME}/qmri_fit ]; then mv ${QMRI_FIT_HOME}/qmri_fit ${QMRI_FIT_HOME}/bin/qmri_fit; fi && \
        if [ -f ${QMRI_FIT_HOME}/bin/qmri-fit ] && [ ! -e ${QMRI_FIT_HOME}/bin/qmri_fit ]; then ln -s qmri-fit ${QMRI_FIT_HOME}/bin/qmri_fit; fi && \
        if [ -f ${QMRI_FIT_HOME}/bin/qmri_fit ] && [ ! -e ${QMRI_FIT_HOME}/bin/qmri-fit ]; then ln -s qmri_fit ${QMRI_FIT_HOME}/bin/qmri-fit; fi && \
        chmod -R a+rX ${QMRI_FIT_HOME} && \
        chmod a+rx ${QMRI_FIT_HOME}/bin/* || true && \
        echo "Installed qmri-fit from ${QMRI_FIT_SRC} to ${QMRI_FIT_HOME}" && \
        ${QMRI_FIT_HOME}/bin/qmri-fit --help >/dev/null 2>&1 || true; \
    else \
        echo "No local qmri-fit binary found under ${QMRI_FIT_SRC}; skipping qmri-fit install."; \
    fi && \
    rm -rf /opt/container-assets

# --------------------------------------------------------------------------------
# 8. qmri-neuropipe and Python dependencies
# --------------------------------------------------------------------------------
WORKDIR /app
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md /app/
COPY src /app/src

RUN ${CONDA_DIR}/bin/python -m pip install --upgrade pip "setuptools>=68,<82" wheel && \
    ${CONDA_DIR}/bin/python -m pip install --no-build-isolation --prefer-binary ".[all]" && \
    ${CONDA_DIR}/bin/python -c "import dmipy_fit; import pkg_resources" && \
    mkdir -p "${DIPY_HOME}" && \
    ${CONDA_DIR}/bin/python -c "from dipy.data import fetch_synb0_weights; fetch_synb0_weights()" && \
    chmod -R a+rX "${DIPY_HOME}"

# --------------------------------------------------------------------------------
# 9. Runtime wrapper
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
