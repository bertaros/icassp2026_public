#!/bin/bash

set -x # Prints each command to the terminal as it is executed

# Distributions
outdir='/pool01/projects/abante_lab/snDGM/icassp2026/results/'
mkdir -p "$outdir"

cd /pool01/code/projects/abante_lab/snDGM/icassp2026_public/gen_mod_ca_img/

# common parameters
seeds=($(seq 1 10)) # doing just the first 10 seeds for now
fluo_noises=(0.5 1.0 1.5 2.0)
latent_dims=(4 8 32 64 128)
vae='SupMlpGpVAE'

# Loop over seed values
for seed in "${seeds[@]}"
do
    for latent_dim in "${latent_dims[@]}"
    do
        for fluo_noise in "${fluo_noises[@]}"
        do
            python train_dgm.py --vae "$vae" --seed "$seed" --fluo_noise "$fluo_noise" --outdir "$outdir" --latent_dim "$latent_dim"
        done
    done
done
