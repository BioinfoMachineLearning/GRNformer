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
# Create Conda environment if it doesn't exist (check by name in this install)
if ! conda env list | awk 'NR>2 && $1=="grnformer" { exit 0 } END { exit 1 }'; then
  conda env create -f environment.yml -n grnformer
fi
conda activate grnformer

# Ensure this Miniconda is used in new shells so "conda activate grnformer" works by name
CONDA_INIT="source \"$HOME/miniconda/etc/profile.d/conda.sh\""
if [[ -f "$HOME/.bashrc" ]] && ! grep -q "miniconda/etc/profile.d/conda.sh" "$HOME/.bashrc"; then
  echo "" >> "$HOME/.bashrc"
  echo "# GRNformer: use this Miniconda so 'conda activate grnformer' works" >> "$HOME/.bashrc"
  echo "$CONDA_INIT" >> "$HOME/.bashrc"
  echo "Added Miniconda init to ~/.bashrc"
fi
echo "Setup complete. In new terminals use: conda activate grnformer"
echo "In this shell you are already in the grnformer env."

