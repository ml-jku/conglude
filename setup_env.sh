#!/usr/bin/env bash
set -e  # exit on first error

# Name of your environment
ENV_NAME="conglude"

# Create env with correct Python version
conda create -y -n $ENV_NAME python=3.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install heavy binary dependencies via conda
conda install -y -c pytorch -c nvidia pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=12.1    
conda install -y -c conda-forge rdkit=2024.03.5
conda install -y -c pyg -c conda-forge pyg=2.4.0

# Install libgcc-ng and make sure Python uses conda's libstdc++ first
conda install -y libgcc-ng
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# Install the rest of your project from pyproject.toml via pip
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install -e .

echo "Environment '$ENV_NAME' created and ready!"