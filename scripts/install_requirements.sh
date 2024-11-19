#!/bin/bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    cmake \
    build-essential \
    libblas-dev \
    liblapack-dev \
    gfortran

# Create virtual environment
python3 -m venv llm_env
source llm_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install \
    transformers \
    flask \
    requests \
    numpy \
    psutil \
    gputil \
    logging \
    accelerate \
    bitsandbytes \
    safetensors
