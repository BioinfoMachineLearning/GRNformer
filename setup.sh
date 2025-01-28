#!/bin/bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
export PATH="$HOME/miniconda/bin:$PATH"
# Create a Conda environment
conda env create -f environment.yml
# Activate the Conda environment
source activate grnformer
# Install PyTorch with CUDA 11

