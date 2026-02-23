#!/bin/bash

set -x # Prints each command to the terminal as it is executed

# Distributions
outdir='/pool01/projects/abante_lab/snDGM/mlcb_2025/results/real_data/'
mkdir -p "$outdir"

# common parameters
latent_dims=(8 16 32 64)
vaes=('FixedVarMlpVAE' 'FixedVarSupMlpVAE' 'FixedVarSupMlpDenVAE' 'LearnedVarSupMlpVAE')

# Loop over seed values
for vae in "${vaes[@]}"
do
    for latent_dim in "${latent_dims[@]}"
    do
            python train_dgm_real_data.py --vae "$vae" --seed "$seed" --outdir "$outdir" --latent_dim "$latent_dim"
    done
done
