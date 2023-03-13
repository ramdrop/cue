#!/bin/sh

SERVER=${2:-local} 

if [[ $SERVER = *local* ]]; then
    echo "[FPT INFO] Running on Local: You should manually load modules..."
    conda init zsh
    source /opt/conda/etc/profile.d/conda.sh # you may need to modify the conda path.
    export CUDA_HOME=/usr/local/cuda-11.1
else
    echo "[FPT INFO] Running on Server..."
    conda init bash
    source ~/anaconda3/etc/profile.d/conda.sh

    module purge
    module load autotools 
    module load prun/1.3 
    module load gnu8/8.3.0 
    module load singularity
    
    module load cuDNN/cuda/11.1/8.0.4.30 
    module load cuda/11.1
    module load nccl/cuda/11.1/2.8.3

    echo "[FPT INFO] Loaded all modules."
fi;