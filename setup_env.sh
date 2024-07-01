#!/bin/bash

# Stop the script on any error
set -e

# Check for Conda installation and initialize Conda in script
if [ -z "$(which conda)" ]; then
    echo "Conda is not installed or not in the PATH"
    exit 1
fi

# Attempt to find Conda's base directory and source it (required for `conda activate`)
CONDA_BASE=$(conda info --base)
PATH="${CONDA_BASE}/bin/":$PATH
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create environment and activate it
conda env create -f conda_env_mining.yml
conda activate three-gen-mining
conda info --env
CUDA_HOME=${CONDA_PREFIX}

# echo -e "\n\n[INFO] Installing diff-gaussian-rasterization package\n"
# mkdir -p ./extras/diff_gaussian_rasterization/third_party
# git clone --branch 0.9.9.0 https://github.com/g-truc/glm.git ./extras/diff_gaussian_rasterization/third_party/glm
# pip install ./extras/diff_gaussian_rasterization

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
cd third_party/diff-gaussian-rasterization &&  pip install -e .
cd ..
cd ..
cd third_party/generative_models &&  pip install -e .
cd ..
cd ..
cd third_party/segmentation_models &&  pip install -e .
cd ..
cd ..

mkdir checkpoints && cd checkpoints
wget https://huggingface.co/camenduru/GRM/resolve/main/grm_u.pth
# wget https://huggingface.co/camenduru/GRM/resolve/main/instant3d.pth
wget https://huggingface.co/camenduru/sv3d/resolve/main/sv3d_p.safetensors
cd ..
# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Generate the generation.config.js file for PM2 with specified configurations
cat <<EOF > generation.config.js
module.exports = {
  apps : [{
    name: 'generation',
    script: 'serve.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
  }]
};
EOF

echo -e "\n\n[INFO] generation.config.js generated for PM2."