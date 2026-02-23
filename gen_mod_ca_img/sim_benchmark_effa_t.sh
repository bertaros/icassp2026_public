#!/bin/bash

set -x # Prints each command to the terminal as it is executed

# Distributions
outdir='/pool01/projects/abante_lab/snDGM/mlcb_2025/results/benchmark_unimodal/round_2/'
mkdir -p "$outdir"

cd /pool01/code/projects/abante_lab/snDGM/mlcb2025_public/gen_mod_ca_img/

# common parameters
seeds=($(seq 1 10))
latent_dims=(4 8 32 64 128)
fluo_noises=(0.5 1.0 1.5 2.0)

# Loop over seed values
for seed in "${seeds[@]}"
do
    # Loop over latent_dim dimensions
    for latent_dim in "${latent_dims[@]}"
    do
        # Parallelize over fluo_noise values
        for fluo_noise in "${fluo_noises[@]}"
        do
            python train_effa.py --seed "$seed" --fluo_noise "$fluo_noise" --outdir "$outdir" --latent_dim "$latent_dim"
        done
    done
done
