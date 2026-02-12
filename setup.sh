#!/bin/bash
set -e
# Download and install Miniconda (skip if already present)
if [[ ! -d "$HOME/miniconda" ]]; then
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p "$HOME/miniconda"
  rm miniconda.sh
fi
export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"
# Create Conda environment if it doesn't exist
if ! conda env list | grep -qw grnformer; then
  conda env create -f environment.yml -n grnformer
fi
conda activate grnformer
echo "Setup complete. Use: conda activate grnformer"

