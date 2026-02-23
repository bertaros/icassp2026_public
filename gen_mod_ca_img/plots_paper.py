#%%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import sem, t


import os
import umap
import time
import psutil
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Pyro
import pyro
import pyro.distributions as dist

# PyTorch
import torch
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Our modules
from ca_sn_gen_models.models import FA
from ca_sn_gen_models.utils import superprint
from ca_sn_gen_models.evaluation import kBET
from ca_sn_gen_models.evaluation import get_hdbscan_ari
from ca_sn_gen_models.evaluation import evaluate_latent_svm
from ca_sn_gen_models.evaluation import compute_local_entropy

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superprint(f"Using device: {device}")

# Set the style for seaborn
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['axes.titlepad'] = 10
plt.rcParams['axes.labelpad'] = 5
plt.rcParams['font.family'] = 'Helvetica'
sns.set_context("notebook")
sns.set_style("ticks")

# os.chdir('/pool01/projects/abante_lab/snDGM/icassp2026/')

from benchmark import scatter_with_ci

#%%
# Import data for plots 2,3


base_results_dir = '/pool01/projects/abante_lab/snDGM/mlcb_2025/results/'
# Load data
results_dir_1 = '/pool01/projects/abante_lab/snDGM/mlcb_2025/results/benchmark_unimodal/round_2/'
results_dir_2 = '/pool01/projects/abante_lab/snDGM/mlcb_2025/results/benchmark_unimodal/gpvae'
results_dir_3 = '/pool01/projects/abante_lab/snDGM/mlcb_2025/results/benchmark_unimodal'


# TODO: get from round 2!
with open(os.path.join(results_dir_1, 'results_summary_BFA.txt'), 'r') as f:
    df_effa_t = pd.read_csv(f, sep='\t')
    df_effa_t['Model'] = 'BFA'

with open(os.path.join(base_results_dir, 'round_3/results_summary_effa_time_test_group_23_no_FS.txt'), 'r') as f:
    df_effa_g23 = pd.read_csv(f, sep='\t')
    df_effa_g23['Model'] = 'BFA'

with open(os.path.join(base_results_dir, 'round_3/results_summary_effa_time_test_group_235_no_FS_ok.txt'), 'r') as f:
    df_effa_g235 = pd.read_csv(f, sep='\t')
    df_effa_g235['Model'] = 'BFA'


## VAE
with open(os.path.join(base_results_dir, 'round_3/results_summary_model_FixedVarMlpVAE_test_group_23_no_FS.txt'), 'r') as f:
    df_vae_g23 = pd.read_csv(f, sep='\t')
    df_vae_g23['Model'] = 'VAE'

with open(os.path.join(base_results_dir, 'round_3/results_summary_model_FixedVarSupMlpVAE_test_group_23_no_FS.txt'), 'r') as f:
    df_svae_g23 = pd.read_csv(f, sep='\t')
    df_svae_g23['Model'] = 'SVAE'

with open(os.path.join(base_results_dir, 'round_3/results_summary_model_SupMlpGpVAE_test.txt'), 'r') as f:
    df_gpvae_g23 = pd.read_csv(f, sep='\t')
    df_gpvae_g23['Model'] = 'GPVAE'

# For each latent dim and seed, keep best value of num_ind_pts based on kbet_test
df_time_SupMlpGpVAE_g23_filt = df_gpvae_g23.loc[df_gpvae_g23.groupby(['latent_dim', 'seed', 'fluo_noise'])['mae_x_test'].idxmin()].reset_index(drop=True)

df_group_23 = pd.concat([df_effa_g23, df_vae_g23, df_svae_g23, df_time_SupMlpGpVAE_g23_filt], axis=0).reset_index(drop=True)

# order models
df_group_23['Model'] = pd.Categorical(
    df_group_23['Model'], 
    categories=['BFA', 'VAE', 'SVAE', 'GPVAE'], 
    ordered=True
)

# filter out noise levels 0 and 2.5
df_group_23 = df_group_23[df_group_23['fluo_noise'].isin([0.5, 1.0, 1.5, 2.0])]

# latent dims
df_group_23 = df_group_23[df_group_23['latent_dim'].isin([4, 8, 16, 32, 128, 256])]
lat_levels = df_group_23['latent_dim'].unique()

noise_level = 1.5

# fig 2b mlcb 2025
df_fig2 = df_group_23[df_group_23['fluo_noise'].isin([noise_level])] # .isin([1])]

df_fig2 = df_fig2[df_fig2['Model'].isin(['BFA', 'VAE', 'SVAE', 'GPVAE'])]
df_fig2['Model'] = pd.Categorical(df_fig2['model'], categories=['BFA', 'VAE', 'SVAE', 'GPVAE'], ordered=True)

metric = 'kbet_test'
best_latent_dims = {}
for model in df_group_23['Model'].unique():
    df_model = df_group_23[df_group_23['Model'] == model]
    best_latent_dims[model] = {}
    for noise in df_model['fluo_noise'].unique():
        df_noise = df_model[df_model['fluo_noise'] == noise]
        mean_kbet = df_noise.groupby('latent_dim')[metric].mean()
        best_latent_dim = mean_kbet.idxmin()  # Get the latent_dim with the lowest mean kbet_val
        best_latent_dims[model][noise] = best_latent_dim

# Do a single boxplot of kbet_val for each model, using only the best latent dimension for each noise level, color by noise level
df_best = pd.DataFrame()
for model in df_group_23['Model'].unique():
    for noise in df_group_23['fluo_noise'].unique():
        best_latent_dim = best_latent_dims[model][noise]
        df_subset = df_group_23[(df_group_23['Model'] == model) & (df_group_23['fluo_noise'] == noise) & (df_group_23['latent_dim'] == best_latent_dim)]
        df_best = pd.concat([df_best, df_subset], axis=0)
# %%
##############################################################################################################
# Forward pass for UMAP plots
##############################################################################################################

################################################################################
# EFFA
################################################################################
seed = 1
lr = 0.01
norm = 'log1p'
orthog = False
retrain = False
domain = 'time'
latent_dim = 4
fluo_noise = 1.0
num_epochs = 2000
save_model = False
target_dist = getattr(dist, 'Normal')
outdir = '/pool01/projects/abante_lab/snDGM/mlcb_2025/results/round_3/'


superprint('Reading in data...')

# included groups
group_set = [2, 3]
group_set_2 = []
sample_set = [0, 1, 2]

# directory
data_dir = '/pool01/projects/abante_lab/snDGM/mlcb_2025/data/caimg/normalized/'
data_dir_2 = '/pool01/projects/abante_lab/snDGM/sims_brb_spring_2025/data/rolled/'

# metadata
metadata_list = []
for group in group_set:
    for sample in sample_set:
        metadata_path_sims = f'{data_dir}/params_group_{group}_sample_{sample}.tsv.gz'
        metadata_sims = pd.read_csv(metadata_path_sims, sep='\t')
        metadata_list.append(metadata_sims)

for group in group_set_2:
    for sample in sample_set:
        metadata_path_sims = f'{data_dir_2}/params_group_{group}_sample_{sample}.tsv.gz'
        metadata_sims = pd.read_csv(metadata_path_sims, sep='\t')
        metadata_list.append(metadata_sims)

# add excitatory and inhibitory labels
ei = 200 * ['E']  + 800 * ['I']
ei_meta = ei * 9

# concatenate dataframes
meta_df = pd.concat(metadata_list, axis=0, ignore_index=True)

# create label combining group and sample
firing_labels = meta_df['firing_type'].values
group_labels = meta_df['group'].values
sample_labels = meta_df['sample'].values
group_sample_labels = np.array([f'{g}_{s}' for g, s in zip(group_labels, sample_labels)])

# Encode group_sample_labels using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(group_sample_labels)
encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)

# Convert to one-hot encoding
num_classes = len(np.unique(encoded_labels))
oh_encoded_labels = torch.nn.functional.one_hot(encoded_labels,num_classes=num_classes)

# traces
data_list = []
for group in group_set:
    for sample in sample_set:
        data_path_sims = f'{data_dir}/sigma_{fluo_noise}/fluo_group_{group}_sample_{sample}.tsv.gz'
        data_sims = pd.read_csv(data_path_sims, sep='\t', header=None).values
        data_list.append(torch.tensor(data_sims, dtype=torch.float32))

for group in group_set_2:
    for sample in sample_set:
        data_path_sims = f'{data_dir_2}/sigma_{fluo_noise}/fluo_group_{group}_sample_{sample}.tsv.gz'
        data_sims = pd.read_csv(data_path_sims, sep='\t', header=None).values
        data_list.append(torch.tensor(data_sims, dtype=torch.float32))

# concatenate data
x = torch.cat(data_list, dim=0)

# subsample data
x = x[:,:10000]

# compute average signal for all xs
x_mean = x.mean(dim=1, keepdim=True)

# do real FFT of the data
xfft = torch.fft.rfft(x, axis=1)

# get amplitude and phase
a = torch.abs(xfft)
p = torch.angle(xfft)

# Eliminate FS neurons
meta_df = meta_df[meta_df['firing_type'] != 'FS']
x = x[meta_df.index]
oh_encoded_labels = oh_encoded_labels[meta_df.index]
firing_labels = firing_labels[meta_df.index]
group_labels = group_labels[meta_df.index]
sample_labels = sample_labels[meta_df.index]
encoded_labels = encoded_labels[meta_df.index]
x_mean = x_mean[meta_df.index]
superprint(f'Loaded {len(x)} traces with {x.shape[1]} time points each after removing FS neurons.')

meta_df = meta_df.reset_index(drop=True)


superprint('Normalizing data...')

if domain == 'time':
    superprint('(Not) normalizing data in time domain...')
    data = x
    norm = None
    # data = normalize_data(x)
else:
    superprint('Normalizing data in frequency domain...')
    # normalize the data
    data = torch.log1p(a)

    # plot histogram of a and ascaled
    plt.hist(data.flatten(), bins=100, alpha=0.5, label='Normalized', color='orange')
    plt.title('Amplitude')
    plt.legend()
    plt.show()

# split and create data loader
train_val_indices, test_indices = train_test_split(np.arange(len(data)), test_size=0.1, random_state=seed)
train_indices, val_indices = train_test_split(train_val_indices, test_size=0.11, random_state=seed)
train_data = data[train_indices]
val_data = data[val_indices]
test_data = data[test_indices]

# create datasets
train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)
test_dataset = TensorDataset(test_data)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=train_data.shape[0], shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=val_data.shape[0], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=test_data.shape[0], shuffle=False)


# EFFA model

# model path
model_path = f'{outdir}/models/BFA_seed_{seed}_latent_{latent_dim}_fluo_{fluo_noise}_test_group_23_no_FS.pt'

# Initialize the FA model
effa = FA(data.shape[1], latent_dim, like=target_dist, domain=domain, device=device)

superprint(f'Model already exists at {model_path}. Loading model...')
# effa.load_state_dict(torch.load(model_path, map_location=device))
checkpoint = torch.load(model_path, map_location=device)
effa.load_state_dict(checkpoint["model_state"])
pyro.get_param_store().set_state(checkpoint["pyro_params"])
effa.to(device)

# Get embedding

# get posterior values of Z, W, and sigma
zloc_tr, W_loc, sigma_loc = effa.get_posterior()

pyro_params = {}
for name, value in pyro.get_param_store().items():
    pyro_params[name] = value.detach().cpu()

W_loc = pyro_params['W_loc']
sigma_loc = pyro_params['sigma_loc']
Sm = torch.diag(1.0 / sigma_loc) # Sm.shape
A = W_loc @ Sm @ W_loc.T
Am = torch.linalg.inv(A + torch.eye(A.shape[0], device=A.device))
Hhat = Sm @ W_loc.T @ Am

zloc_test = test_data @ Hhat


# Perform UMAP dimensionality reduction
reducer = umap.UMAP(n_components=2)
Z_umap_effa = reducer.fit_transform(zloc_tr)

umap_df_effa = pd.DataFrame(Z_umap_effa, columns=['UMAP1', 'UMAP2'])
umap_df_effa['Group'] = group_labels[train_indices]
umap_df_effa['Sample'] = sample_labels[train_indices]
umap_df_effa['Firing'] = firing_labels[train_indices]
umap_df_effa['Model'] = 'BFA'

################################################################################
# SVAE 
################################################################################

from ca_sn_gen_models.models import FixedVarSupMlpVAE as vae_model 
batch_size = 1000
# split labels
train_labels = oh_encoded_labels[train_indices]
val_labels = oh_encoded_labels[val_indices]
test_labels = oh_encoded_labels[test_indices]

# create datasets
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
test_dataset = TensorDataset(test_data, test_labels)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


vae = 'FixedVarSupMlpVAE'

model_type = 'Supervised'
out_var = 'Fixed'
mask = False
# model path
model_path = f'{outdir}/models/{vae}_seed_{seed}_latent_{latent_dim}_fluo_{fluo_noise}_test_group_23_no_FS.pt'

# init model
superprint(f'Initializing model {vae} with latent dimension {latent_dim} and noise level {fluo_noise}...')

# Initialize model with num_classes
model = vae_model(data.shape[1], latent_dim, num_classes, device=device)
superprint(f'Model already exists at {model_path}. Loading model...')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)


superprint('Getting posterior estimates...')

# Set model to evaluation mode
model.eval()

# get posterior value of Z given X and Y for training data
zloc_tr = []
xhat_tr = []

for batch in train_loader:

    x_batch = batch[0].to(device=device)
    y_batch = batch[1].to(device=device)
    
    with torch.no_grad():
        
        if mask:
            # For masked models, we need to pass the mask as well
            zloc_tr_batch, _ = model.encode(x_batch, y_batch, torch.ones_like(x_batch, dtype=torch.float32))
        else:
            # For non-masked models, we can just pass x_batch and y_batch
            zloc_tr_batch, _ = model.encode(x_batch, y_batch)
        zloc_tr.append(zloc_tr_batch.detach().cpu())
        
        if out_var == 'Fixed':
            xhat_tr_batch = model.decode(zloc_tr_batch, y_batch)
        else:  # Learned
            xhat_tr_batch,_ = model.decode(zloc_tr_batch, y_batch)
        
        xhat_tr.append(xhat_tr_batch.detach().cpu())

# concatenate batches
zloc_tr = torch.cat(zloc_tr, dim=0)
xhat_tr = torch.cat(xhat_tr, dim=0)


# Perform UMAP dimensionality reduction
reducer = umap.UMAP(n_components=2)
Z_umap_svae = reducer.fit_transform(zloc_tr)

umap_df_svae = pd.DataFrame(Z_umap_svae, columns=['UMAP1', 'UMAP2'])
umap_df_svae['Group'] = group_labels[train_indices]
umap_df_svae['Sample'] = sample_labels[train_indices]
umap_df_svae['Firing'] = firing_labels[train_indices]
umap_df_svae['Model'] = 'SVAE'

# Chnage group 2 to 1 and group 3 to 2
umap_df_effa['Group'] = umap_df_effa['Group'].replace({2: 1, 3: 2})
umap_df_svae['Group'] = umap_df_svae['Group'].replace({2: 1, 3: 2})

#%%
########################################################################################################
# PLOT
########################################################################################################
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

fontsize = 16

# Create figure with outer grid
fig = plt.figure(figsize=(22, 12))
outer = gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.25)

# --- Top-left: subdivide into two rows --- #

gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 0], hspace=0.23)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])


trace_RS_G2 = x[(meta_df['firing_type'] == 'RS') & (meta_df['group'] == 2)][np.random.randint(0,199)]
trace_LTS_G2 = x[(meta_df['firing_type'] == 'LTS') & (meta_df['group'] == 2)][np.random.randint(0,799)]
trace_RS_G3 = x[(meta_df['firing_type'] == 'RS') & (meta_df['group'] == 3)][np.random.randint(0,199)]
trace_LTS_G3 = x[(meta_df['firing_type'] == 'LTS') & (meta_df['group'] == 3)][np.random.randint(0,799)]


# Example signals
tvec = np.linspace(0, 10000, 10000)
ax1.plot(tvec, trace_RS_G2, label="RS", alpha=0.7)
ax1.plot(tvec, trace_LTS_G2, label="LTS", alpha=0.7)
ax1.set_xlim([0, 10000])
ax1.tick_params(labelsize=fontsize)
ax1.set_title(ax1.get_title(), fontsize=fontsize)
ax1.set_xlabel(ax1.get_xlabel(), fontsize=fontsize)
ax1.legend(fontsize=fontsize)
# Remove ylabel from individual axes
ax1.set_ylabel('')
# ax1.set_ylabel(ax1.get_ylabel(), fontsize=fontsize)

ax2.plot(tvec, trace_RS_G3, alpha=0.7)
ax2.plot(tvec, trace_LTS_G3, alpha=0.7)
ax2.set_xlabel('Simulated time')
ax2.set_xlim([0, 10000])
ax2.tick_params(labelsize=fontsize)
ax2.set_title(ax2.get_title(), fontsize=fontsize)
ax2.set_xlabel(ax2.get_xlabel(), fontsize=fontsize)
ax2.set_ylabel('')  # Remove ylabel from ax2

# Add shared ylabel between the two plots
fig.text(
    0.09, 0.72, r'$\mathbf{x}_{s,n}$',
    va='center', ha='center', rotation='vertical', fontsize=25
)


# --- Top-right: kBET scatter plot --- #

ax3 = fig.add_subplot(outer[0, 1])

metric_x = 'sil_test'
metric_y = 'kbet_test'
conf = 0.95

grouped = df_group_23.groupby(['Model', 'latent_dim'], observed=True)
df_mean = grouped.mean(numeric_only=True).reset_index()

# Compute standard error and confidence intervals
n = grouped.size().values
se_x = grouped[metric_x].sem().values
se_y = grouped[metric_y].sem().values
h_x = se_x * t.ppf((1 + conf) / 2., n - 1)
h_y = se_y * t.ppf((1 + conf) / 2., n - 1)

df_mean[f'{metric_x}_ci'] = h_x
df_mean[f'{metric_y}_ci'] = h_y

# Scatter plot with error bars
sns.scatterplot(
    data=df_mean,
    x=metric_x,
    y=metric_y,
    hue='Model',
    style='latent_dim',
    palette="pastel",
    s=250,
    alpha=0.8,
    ax=ax3
)

# Add error bars
for _, row in df_mean.iterrows():
    ax3.errorbar(
        row[metric_x], row[metric_y],
        xerr=row[f'{metric_x}_ci'], yerr=row[f'{metric_y}_ci'],
        fmt='none', ecolor='gray', alpha=0.7, capsize=3, zorder=0
    )

ax3.set_xlabel('')  # Remove ylabel from axis
fig.text(
    0.71, 0.49, 'Silhouette Score',
    va='center', ha='center', fontsize=fontsize
)
ax3.set_ylabel('')  # Remove ylabel from axis

# Add shared ylabel, separated from axis
fig.text(
    0.52, 0.7, 'kBET Value',
    va='center', ha='center', rotation='vertical', fontsize=fontsize
)
ax3.tick_params(labelsize=fontsize)
ax3.legend(bbox_to_anchor=(0.72, 1), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)



# --- Bottom-left: subdivide into two columns --- #

gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0], wspace=0.2)
ax4 = fig.add_subplot(gs2[0])
ax5 = fig.add_subplot(gs2[1])

# Add a group_sample column to both dataframes
umap_df_effa['Group_Sample'] = umap_df_effa['Group'].astype(str) + '_' + umap_df_effa['Sample'].astype(str)
umap_df_svae['Group_Sample'] = umap_df_svae['Group'].astype(str) + '_' + umap_df_svae['Sample'].astype(str)

# Change from 1_1 to 1, 1_2 to 2, ..., 2_2 to 5, 2_3 to 6
group_sample_mapping = {
    '1_0': '1',
    '1_1': '2',
    '1_2': '3',
    '2_0': '4',
    '2_1': '5',
    '2_2': '6'
}   
umap_df_effa['Group_Sample'] = umap_df_effa['Sample_idx'].map(group_sample_mapping)
umap_df_svae['Group_Sample'] = umap_df_svae['Sample_idx'].map(group_sample_mapping)

# Make colormap from viridis with as many colors as unique group_sample combinations
unique_groups = umap_df_effa['Sample_idx'].unique()
num_colors = len(unique_groups)
cmap = plt.get_cmap('tab20', num_colors)
color_dict = {group: cmap(i) for i, group in enumerate(sorted(unique_groups))}  # Sort groups

# UMAP plots
sns.scatterplot(
    data=umap_df_effa, x='UMAP1', y='UMAP2', hue='Sample_idx', style='Firing',
    palette=color_dict, alpha=0.7, s=100, ax=ax4
)
ax4.set_title('BFA', fontsize=fontsize)
# ax4.legend.remove()
sns.scatterplot(
    data=umap_df_svae, x='UMAP1', y='UMAP2', hue='Sample_idx', style='Firing',
    palette=color_dict, alpha=0.7, s=100, ax=ax5
)
ax5.set_title('SVAE', fontsize=fontsize)
ax4.tick_params(labelsize=fontsize)
ax4.set_xlabel(ax4.get_xlabel(), fontsize=fontsize)
ax4.set_ylabel(ax4.get_ylabel(), fontsize=fontsize)
ax5.legend(bbox_to_anchor=(0.5, 1), loc='upper right', fontsize=12, handletextpad=0.1)
ax5.set_ylabel('')
ax5.tick_params(labelsize=fontsize)
ax5.set_xlabel(ax5.get_xlabel(), fontsize=fontsize)


# --- Bottom-right: kBET boxplot --- #

palette = sns.color_palette("viridis", as_cmap=True)
# Map noise levels to colors using the full colormap, but invert the order
noise_levels = sorted(df_best['fluo_noise'].unique())
colors = [palette(i / (len(noise_levels) - 1)) for i in reversed(range(len(noise_levels)))]
color_dict = dict(zip(noise_levels, colors))

ax6 = fig.add_subplot(outer[1, 1])
# Noise level boxplot
sns.boxplot(
    x='Model',
    y='kbet_test',
    hue='fluo_noise',
    data=df_best,
    showfliers=False,
    palette=color_dict,
    ax=ax6,
    linewidth=0.7  # Make boxplot lines thinner
)
ax6.set_xlabel('Model', fontsize=fontsize)
ax6.set_ylabel('')  # Remove ylabel from axis

# Add shared ylabel, separated from axis
fig.text(
    0.52, 0.28, 'kBET Value',
    va='center', ha='center', rotation='vertical', fontsize=fontsize
)

ax6.tick_params(labelsize=fontsize)
ax6.legend(title='Noise Level', bbox_to_anchor=(0.78, 1), loc='upper left', fontsize=fontsize, title_fontsize=fontsize)

# Put the figure labels
fig.text(0.08, 0.88, 'A', fontsize=30)
fig.text(0.51, 0.88, 'B', fontsize=30)
fig.text(0.08, 0.46, 'C', fontsize=30)
fig.text(0.51, 0.46, 'D', fontsize=30)

plt.savefig(f'/pool01/projects/abante_lab/snDGM/icassp2026/figure_1_icassp_2026_v1.pdf', bbox_inches='tight', dpi=900)

# %%
