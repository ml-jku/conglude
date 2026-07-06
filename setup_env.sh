#!/usr/bin/env bash
set -e  # exit on first error
 
# Name of your environment
ENV_NAME="conglude"
 
# Create env with correct Python version
conda create -y -n $ENV_NAME python=3.11
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME
 
# CUDA wheel tag: pass as first argument, defaults to cu128
tag=${1:-cu128}

# Install heavy binary dependencies
if [ "$tag" = "cu121" ]; then
  pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
  pip install torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
else
  pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/$tag
  pip install torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+$tag.html
fi
conda install -y -c conda-forge rdkit=2024.03.5  # pinned: fingerprint counts differ across versions
 
# Install libgcc-ng and make sure Python uses conda's libstdc++ first
conda install -y libgcc-ng
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
 
# Install the rest of your project from pyproject.toml via pip
pip install -e .
 
echo "Environment '$ENV_NAME' created and ready!"