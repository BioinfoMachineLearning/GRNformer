#!/bin/bash
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
export PATH="$HOME/miniconda/bin:$PATH"
# Create a Conda environment
conda env create -f env.yml
# Activate the Conda environment
source activate grnformer
# Install PyTorch with CUDA 11
conda install -u pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# Install PyTorch Lightning
pip install pytorch-lightning==1.8.6
