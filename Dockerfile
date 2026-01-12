# Dockerfile for qmri-neuropipe
# Based on NVIDIA CUDA 12.2 Runtime on Ubuntu 22.04

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------------
# 1. System Dependencies
# --------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    git \
    build-essential \
    libgl1-mesa-dev \
    libfontconfig1 \
    libfreetype6 \
    libxrender1 \
    libxext6 \
    libsm6 \
    sudo \
    bc \
    dc \
    file \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------------
# 2. Miniconda & MRtrix3 & Python Env
# --------------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda config --add channels conda-forge && \
    conda config --add channels mrtrix3 && \
    conda install -y mrtrix3 python=3.10 pip && \
    conda clean --all --yes

# --------------------------------------------------------------------------------
# 3. FSL (Manual Install)
# --------------------------------------------------------------------------------
# FSL Installer requires Python. We utilize the conda python.
# Using the official installer in quiet mode (-q) to bypass prompts.

ENV FSLDIR=/usr/local/fsl
ENV PATH=$FSLDIR/bin:$PATH
ENV FSLOUTPUTTYPE=NIFTI_GZ

RUN curl -fsSL https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py -o /tmp/fslinstaller.py && \
    python3 /tmp/fslinstaller.py -d $FSLDIR -q --no_env && \
    rm /tmp/fslinstaller.py

# --------------------------------------------------------------------------------
# 4. Convert3D (C3D)
# --------------------------------------------------------------------------------
RUN mkdir -p /opt/c3d && \
    wget https://sourceforge.net/projects/c3d/files/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz/download -O /tmp/c3d.tar.gz && \
    tar -xzf /tmp/c3d.tar.gz -C /opt/c3d --strip-components=1 && \
    rm /tmp/c3d.tar.gz

ENV C3DPATH=/opt/c3d/bin
ENV PATH=$C3DPATH:$PATH

# --------------------------------------------------------------------------------
# 5. qmri-neuropipe
# --------------------------------------------------------------------------------
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip && \
    pip install ".[all]"

ENTRYPOINT ["qmri-neuropipe"]
CMD ["--help"]
