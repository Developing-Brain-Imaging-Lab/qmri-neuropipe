# Dockerfile for qmri-neuropipe
# Based on Ubuntu 22.04 LTS (Jammy Jellyfish)

FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------------
# 1. System Dependencies
# --------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    software-properties-common \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    zlib1g-dev \
    bc \
    dc \
    file \
    libfontconfig1 \
    libfreetype6 \
    libgl1-mesa-dev \
    libgl1-mesa-dri \
    libglu1-mesa-dev \
    libgomp1 \
    libice6 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    libxt6 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------------
# 2. NeuroDebian (FSL, MRtrix3)
# --------------------------------------------------------------------------------
RUN wget -O- http://neuro.debian.net/lists/jammy.us-nh.full | tee /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key adv --recv-keys --keyserver hkps://keyserver.ubuntu.com 0xA5D32F012649A5A9

RUN apt-get update && apt-get install -y \
    fsl-core \
    mrtrix3 \
    && rm -rf /var/lib/apt/lists/*

# FSL Configuration
ENV FSLDIR=/usr/share/fsl/5.0
ENV PATH=$FSLDIR/bin:$PATH
ENV FSLOUTPUTTYPE=NIFTI_GZ
# Provide entrypoint source if needed, but ENV is usually enough for FSL 5.0 in NeuroDebian

# --------------------------------------------------------------------------------
# 3. Convert3D (C3D)
# --------------------------------------------------------------------------------
RUN mkdir -p /opt/c3d && \
    wget https://sourceforge.net/projects/c3d/files/c3d/1.0.0/c3d-1.0.0-Linux-x86_64.tar.gz/download -O /tmp/c3d.tar.gz && \
    tar -xzf /tmp/c3d.tar.gz -C /opt/c3d --strip-components=1 && \
    rm /tmp/c3d.tar.gz

ENV C3DPATH=/opt/c3d/bin
ENV PATH=$C3DPATH:$PATH

# --------------------------------------------------------------------------------
# 4. FreeSurfer
# --------------------------------------------------------------------------------
# NOTE: FreeSurfer is required for Synb0 (mri_convert, etc.) but is very large (~10GB).
# We provide the install commands below. 
# Alternatively, you can mount an external FreeSurfer installation at runtime.
#
# ENV FREESURFER_HOME=/opt/freesurfer
# RUN wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.4.1/freesurfer-linux-ubuntu22_amd64-7.4.1.tar.gz -O /tmp/fs.tar.gz && \
#     tar -xzf /tmp/fs.tar.gz -C /opt && \
#     rm /tmp/fs.tar.gz
# ENV PATH=$FREESURFER_HOME/bin:$PATH

# --------------------------------------------------------------------------------
# 5. Python Environment & qmri-neuropipe
# --------------------------------------------------------------------------------
WORKDIR /app
COPY . /app

# Upgrade pip and install package
RUN python3 -m pip install --upgrade pip && \
    pip install ".[all]"

# --------------------------------------------------------------------------------
# 6. Entrypoint
# --------------------------------------------------------------------------------
# Ensure FSL is sourced? (ENV above should handle it)
ENTRYPOINT ["qmri-neuropipe"]
CMD ["--help"]
