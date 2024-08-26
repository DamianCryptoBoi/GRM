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
conda env create -f conda_env_grm.yml
conda activate grm
conda info --env
CUDA_HOME=${CONDA_PREFIX}

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

cd third_party/diff-gaussian-rasterization &&  pip install -e .

cd ../..

mkdir checkpoints && cd checkpoints

wget https://huggingface.co/camenduru/GRM/resolve/main/grm_r.pth

wget https://huggingface.co/camenduru/GRM/resolve/main/grm_u.pth

wget https://huggingface.co/camenduru/GRM/resolve/main/grm_zero123plus.pth

wget https://huggingface.co/camenduru/GRM/resolve/main/instant3d.pth

cd ..


# Store the path of the Conda interpreter
CONDA_INTERPRETER_PATH=$(which python)

# Generate the generation.config.js file for PM2 with specified configurations
cat <<EOF > generation.config.js
module.exports = {
  apps : [{
    name: 'grm',
    script: 'app.py',
    interpreter: '${CONDA_INTERPRETER_PATH}',
  }]
};
EOF

echo -e "\n\n[INFO] generation.config.js generated for PM2."