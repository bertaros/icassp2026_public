# snDGM: Single-nucleus Deep Generative Models

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the code for the ICASSP 2026 paper on single-nucleus deep generative models (snDGM). The project implements and benchmarks various variational autoencoder (VAE) architectures for analyzing calcium imaging and single-nucleus data.

## Overview

The repository provides implementations for:
- **Multiple VAE architectures**: Fixed/learned variance, supervised, denoising, and GP-based variants
- **Benchmark experiments**: Systematic evaluation across different noise levels and latent dimensions
- **Real data applications**: Training and evaluation on real single-nucleus datasets
- **Evaluation metrics**: kBET, silhouette score, HDBSCAN ARI, SVM accuracy, local entropy

## Repository Structure

```
icassp2026_public/
├── gen_mod_ca_img/          # Simulated data experiments
│   ├── train_dgm.py         # Train deep generative models
│   ├── train_effa.py        # Train expectation-maximization FA
│   ├── benchmark.py         # Benchmark analysis and plotting
│   ├── plots_paper.py       # Generate paper figures
│   ├── sim_benchmark_dgm.sh # Run DGM benchmarks
│   ├── sim_benchmark_effa_t.sh
│   └── sim_benchmark_gpvae.sh
└── real_data/               # Real data experiments
    ├── train_dgm_real_data.py
    └── benchmark_real_data.sh
```

## Installation

### Prerequisites

- Python 3.7+
- CUDA-enabled GPU (recommended)
- Custom package: `ca_sn_gen_models`

### Dependencies

```bash
pip install torch torchvision
pip install pyro-ppl
pip install numpy pandas scipy
pip install scikit-learn
pip install matplotlib seaborn
pip install umap-learn
pip install psutil
```

### Installing ca_sn_gen_models

This project depends on the custom `ca_sn_gen_models` package, which should be installed separately. Contact the authors or check the lab repository for access.

## Usage

### Training Models on Simulated Data

Train a specific VAE model:

```bash
python gen_mod_ca_img/train_dgm.py \
    --vae FixedVarSupMlpVAE \
    --seed 1 \
    --latent_dim 8 \
    --fluo_noise 1.0 \
    --num_epochs 1000 \
    --rate 0.0001 \
    --batch_size 256 \
    --outdir ./results/
```

### Available Models

- `FixedVarMlpVAE`: Standard VAE with fixed decoder variance
- `LearnedVarMlpVAE`: VAE with learned decoder variance
- `FixedVarSupMlpVAE`: Supervised VAE with fixed variance
- `LearnedVarSupMlpVAE`: Supervised VAE with learned variance
- `FixedVarSupMlpDenVAE`: Supervised denoising VAE with fixed variance
- `LearnedVarSupMlpDenVAE`: Supervised denoising VAE with learned variance
- `SupMlpGpVAE`: Supervised GP-VAE

### Running Benchmarks

Run comprehensive benchmark experiments:

```bash
# Benchmark DGM variants
bash gen_mod_ca_img/sim_benchmark_dgm.sh

# Benchmark EFFA models
bash gen_mod_ca_img/sim_benchmark_effa_t.sh

# Benchmark GP-VAE models
bash gen_mod_ca_img/sim_benchmark_gpvae.sh
```

### Training on Real Data

```bash
python real_data/train_dgm_real_data.py \
    --vae FixedVarSupMlpVAE \
    --seed 1 \
    --latent_dim 8 \
    --num_epochs 1000 \
    --rate 0.0001 \
    --batch_size 1000 \
    --outdir ./results/real_data/
```

Or run the full benchmark:

```bash
bash real_data/benchmark_real_data.sh
```

## Command Line Arguments

### Common Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--vae` | str | `FixedVarMlpVAE` | Model architecture to use |
| `--seed` | int | 1 | Random seed for reproducibility |
| `--latent_dim` | int | 8 | Dimensionality of latent space |
| `--num_epochs` | int | 1000 | Number of training epochs |
| `--rate` | float | 0.0001 | Learning rate |
| `--batch_size` | int | 256 | Batch size for training |
| `--outdir` | str | `./output` | Output directory for results |
| `--fluo_noise` | float | 1.0 | Fluorescence noise level (simulated data only) |
| `--num_ind_pts` | int | 1500 | Number of inducing points (GP models only) |

## Evaluation Metrics

The models are evaluated using multiple metrics:

- **kBET**: k-nearest neighbor batch effect test
- **Silhouette Score**: Clustering quality measure
- **HDBSCAN ARI**: Adjusted Rand Index for hierarchical clustering
- **SVM Accuracy**: Linear SVM classification on latent representations
- **Local Entropy**: Information mixing in latent space

## Visualization

Generate paper figures and benchmark plots:

```bash
python gen_mod_ca_img/plots_paper.py
python gen_mod_ca_img/benchmark.py
```

The benchmark script creates:
- Boxplots comparing models across noise levels
- Performance metrics vs. latent dimensions
- Cross-model comparisons

## Hardware Requirements

- **RAM**: 16GB+ recommended
- **GPU**: CUDA-capable GPU with 8GB+ VRAM recommended for faster training
- **Storage**: Sufficient space for model checkpoints and results

## Output Format

Training scripts generate the following outputs:
- Model checkpoints (`.pt` files)
- Training logs and metrics (`.csv` files)
- Latent representations
- Evaluation metrics
- Visualization plots

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{snDGM2026,
  title={Single-nucleus Deep Generative Models for Calcium Imaging Data},
  author={[Authors]},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact the Abante Lab.

## Acknowledgments

This work was conducted at the Abante Lab. The research was supported by [funding sources].

---

**Note**: This is research code. For production use, additional testing and optimization may be required.
