#%%

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
# device = 'cpu'  

def running_in_ipython():
    try:
        get_ipython   # type: ignore
        return True
    except NameError:
        return False

###############################################################################################
# arguments
###############################################################################################
#%%

if running_in_ipython():
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

else:
    # Create the parser
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(
        "--fluo_noise",
        type=float,
        required=True,
        default=1.0,
        help="Noise level (default: 1.0)"
    )

    # optional arguments
    parser.add_argument(
        "-l",
        "--latent_dim", 
        type=int, 
        required=False,
        default=32,
        help="Dimension of latent space (default: 32)"
    )

    parser.add_argument(
        "-e",
        "--epochs", 
        type=int, 
        required=False, 
        default=2000,
        help="Number of training epochs (default: 2000)"
    )

    parser.add_argument(
        "--norm", 
        type=str, 
        required=False, 
        default='none',
        choices=['log1p', 'std', 'scale', 'none'], 
        help="norm method"
    )
    parser.add_argument(
        "-s",
        "--seed", 
        type=int, 
        required=False, 
        default=0,
        help="RNG seed (default: 0)"
    )

    parser.add_argument(
        "-r",
        "--rate", 
        type=float, 
        required=False, 
        default=0.01,
        help="Learning rate (default: 0.01)"
    )

    parser.add_argument(
        "--retrain", 
        type=bool, 
        required=False, 
        default=False,
        help="Whether to retrain the model (default: False)"
    )

    parser.add_argument(
        "--save", 
        type=bool, 
        required=False, 
        default=True,
        help="Whether to save the model (default: True)"
    )

    parser.add_argument(
        "--target_dist", 
        type=str, 
        required=False, 
        default='Normal',
        choices=['Normal', 'Laplace', 'Gamma', 'LogNormal'],
        help="likelihood distribution (default: Normal)"
    )

    parser.add_argument(
        "--domain",
        type=str,
        required=False,
        default='time',
        choices=['time', 'frequency'],
        help="Domain of the data (default: time)"
    )

    parser.add_argument(
        '--outdir', 
        type=str, 
        required=False,
        default='./output', 
        help='Folder to save output files'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    seed = args.seed              # seed=1   
    retrain = args.retrain        # retrain=True
    save_model = args.save        # save=False
    norm = args.norm              # norm='log1p'
    latent_dim = args.latent_dim  # latent_dim=8
    num_epochs = args.epochs      # num_epochs=200
    lr = args.rate                # learning_rate=0.005
    fluo_noise = args.fluo_noise  # fluo_noise=1.0
    domain = args.domain          # domain = 'time'
    outdir = args.outdir          # outdir = './output'
    target_dist = getattr(dist, args.target_dist)           # likelihood distribution (default: Normal)

#########################################################################################################
# Testing
#########################################################################################################
#%%

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


#########################################################################################################
# NORMALIZE AND SPLIT DATA
#########################################################################################################

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

#########################################################################################################
# TRAIN
#########################################################################################################
#%%

# model path
model_path = f'{outdir}/models/BFA_seed_{seed}_latent_{latent_dim}_fluo_{fluo_noise}_test_group_23_no_FS.pt'

# Initialize the FA model
effa = FA(data.shape[1], latent_dim, like=target_dist, domain=domain, device=device)

if not os.path.exists(model_path):
    
    superprint(f'Model does not exist at {model_path}. Starting training...')

    # Start the timer
    start_time = time.time()

    # clear cuda memory
    torch.cuda.empty_cache()

    # Clear Pyro parameters
    pyro.clear_param_store()

    # Train the model
    loss_history = effa.train_model(train_loader, num_epochs=num_epochs, lr=lr, patience=20, min_delta=5e0)

    # Save the model
    # torch.save(effa.state_dict(), model_path)
    torch.save({
    "model_state": effa.state_dict(),
    "pyro_params": pyro.get_param_store().get_state()
    }, model_path)

    superprint(f'Model saved to {model_path}')

    # Plot the training loss
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Monitor peak RAM usage
    process = psutil.Process()

    # Get peak memory usage in MB
    peak_memory = process.memory_info().peak_wset / (1024 ** 2) if hasattr(process.memory_info(), 'peak_wset') else process.memory_info().rss / (1024 ** 2)
    superprint(f"Peak RAM usage: {peak_memory:.2f} MB")

    # stop timer
    end_time = time.time()
    training_time = end_time - start_time
    superprint(f'Training time: {training_time:.2f} seconds')

else:
    
    superprint(f'Model already exists at {model_path}. Loading model...')
    # effa.load_state_dict(torch.load(model_path, map_location=device))
    checkpoint = torch.load(model_path, map_location=device)
    effa.load_state_dict(checkpoint["model_state"])
    pyro.get_param_store().set_state(checkpoint["pyro_params"])
    effa.to(device)

    # print pyro parameters
    for name, value in pyro.get_param_store().items():
        superprint(f'{name}: {value}')
    superprint('Model loaded.')


#########################################################################################################
# Clean data
#########################################################################################################

superprint('Reading in zero noise data...')

# traces
data_list = []
for group in group_set:
    for sample in sample_set:
        data_path_sims = f'{data_dir}/sigma_0.0/fluo_group_{group}_sample_{sample}.tsv.gz'
        data_sims = pd.read_csv(data_path_sims, sep='\t', header=None).values
        data_list.append(torch.tensor(data_sims, dtype=torch.float32))

for group in group_set_2:
    for sample in sample_set:
        data_path_sims = f'{data_dir_2}/sigma_0.0/fluo_group_{group}_sample_{sample}.tsv.gz'
        data_sims = pd.read_csv(data_path_sims, sep='\t', header=None).values
        data_list.append(torch.tensor(data_sims, dtype=torch.float32))

# concatenate data
x0 = torch.cat(data_list, dim=0)

# subsample data
x0 = x0[:,:10000]

# Eliminate FS neurons
x0 = x0[meta_df.index]

# split into train and validation
x0_train = x0[train_indices]
x0_val = x0[val_indices]

# FFT
x0_train_fft = torch.fft.rfft(x0_train, axis=1)
x0_val_fft = torch.fft.rfft(x0_val, axis=1)

# get amplitude and phase
a0_train = torch.abs(x0_train_fft)
a0_val = torch.abs(x0_val_fft)
p0_train = torch.angle(x0_train_fft)
p0_val = torch.angle(x0_val_fft)

#########################################################################################################
# EMBEDDING
#########################################################################################################

superprint('Getting posterior estimates...')

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
Z_umap = reducer.fit_transform(zloc_tr)

# Create a DataFrame for easier plotting with seaborn
umap_df = pd.DataFrame({
    'UMAP1': Z_umap[:, 0],
    'UMAP2': Z_umap[:, 1],
    'Group': group_labels[train_indices],
    'Sample': sample_labels[train_indices],
    'FiringType': firing_labels[train_indices],
    'MeanFiring': x_mean[train_indices].squeeze(1).cpu().numpy(),
    "Group_sample": group_sample_labels[train_indices]
})

# Plot using seaborn for Group and Sample
plt.figure(figsize=(8, 5))
scatter = sns.scatterplot(
    data=umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='Group',
    style='Sample',
    palette='Set1',
    s=50
)
scatter.set_title("UMAP of Latent Variables Z (Colored by Group and Sample)")
scatter.set_xlabel("UMAP Dimension 1")
scatter.set_ylabel("UMAP Dimension 2")
plt.legend(title="Groups and Samples", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Plot using seaborn for Firing Type
plt.figure(figsize=(6, 5))
scatter = sns.scatterplot(
    data=umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='FiringType',
    style='Group',
    palette='Set2',
    s=50
)
scatter.set_title("UMAP of Latent Variables Z (Colored by Firing Type)")
scatter.set_xlabel("UMAP Dimension 1")
scatter.set_ylabel("UMAP Dimension 2")
plt.legend(title="Firing Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'/pool01/projects/abante_lab/snDGM/icassp2026/umap_effa_{domain}_train_group_23_no_FS_vertical.pdf', bbox_inches='tight', dpi=300)
plt.show()

# Plot using seaborn for Firing Type
plt.figure(figsize=(8, 5))
scatter = plt.scatter(
    umap_df['UMAP1'],
    umap_df['UMAP2'],
    c=umap_df['MeanFiring'],
    cmap='viridis',
    s=5,
    alpha=0.8
)
plt.colorbar(scatter, label='Mean Firing')
plt.title("UMAP of Latent Variables Z (Colored by Mean Firing Rate)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.tight_layout()
plt.show()

# Obtain int labels for FS, RS, LTS
firing_type_mapping = {'FS': 0, 'RS': 1, 'LTS': 2, 'CH': 3}
firing_type_int = np.array([firing_type_mapping[ft] for ft in firing_labels])

# Train a classifier to predict firing type from latent variables
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Initialize classifier
clf = RandomForestClassifier(n_estimators=100, random_state=seed)
# Train on training set
clf.fit(zloc_tr, firing_type_int[train_indices])
# Predict on test set
y_pred = clf.predict(zloc_test)
# Print classification report
print(classification_report(firing_type_int[test_indices], y_pred, target_names=['RS', 'LTS']))
firing_acc = (y_pred == firing_type_int[test_indices]).mean()

# Train a classifier to predict batch label from latent variables
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Initialize classifier
clf = RandomForestClassifier(n_estimators=100, random_state=seed)
# Train on training set
clf.fit(zloc_tr, encoded_labels[train_indices].numpy())
# Predict on test set
y_pred = clf.predict(zloc_test)
# Print classification report
print(classification_report(encoded_labels[test_indices].numpy(), y_pred, target_names=encoded_labels.unique().numpy().astype(str)))
batch_acc = (y_pred == encoded_labels[test_indices].numpy()).mean()

##############################################################################################################
# Reconstruction of X
##############################################################################################################

superprint('Evaluating reconstruction...')

# Set model to evaluation mode
effa.eval()

# get posterior value of Z given X
zloc_val = val_data @ Hhat

if domain == 'time':
            
    # reconstruct data
    xhat = zloc_val @ W_loc

    # set error metrics of amplitude to None
    mae_a = None
    mse_a = None
    
else:
    
    # Revert log1p scaling
    ahat = torch.expm1(zloc_val @ W_loc)
    
    # Calculate reconstruction error metrics
    mae_a = torch.nn.functional.l1_loss(a0_val, ahat, reduction='mean').item()
    mse_a = torch.nn.functional.mse_loss(a0_val, ahat, reduction='mean').item()

    # do ifft using the reconstructed amplitude and original phase
    xhat = ahat * torch.exp(1j * p0_val)
    xhat = torch.fft.irfft(xhat, n=x.shape[1])

# plot a few examples of the data
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
random_indices = np.random.choice(len(val_data), 4, replace=False)
for i, idx in enumerate(random_indices):
    axs[i].plot(x0_val[idx], label='GT', color='green')
    axs[i].plot(xhat[idx], label='Reconstructed FA', color='red', alpha=0.5)
    axs[i].set_title(f'Sample {idx}')
    axs[i].set_xlabel('Frequency')
    axs[i].set_ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()

# compute reconstruction error with noiseless data
mae_x_val = torch.nn.functional.l1_loss(x0_val, xhat, reduction='mean').item()
mse_x_val = torch.nn.functional.mse_loss(x0_val, xhat, reduction='mean').item()

# compute reconstruction error with test data
mae_x_test = torch.nn.functional.l1_loss(test_data, zloc_test @ W_loc, reduction='mean').item()
mse_x_test = torch.nn.functional.mse_loss(test_data, zloc_test @ W_loc, reduction='mean').item()

###############################################################################################
# Evaluation of the latent representation
###############################################################################################

superprint('Evaluating latent representation...')

## 1. silhouette scores

sil_train_firing = silhouette_score(zloc_tr, firing_labels[train_indices])
sil_val_firing = silhouette_score(zloc_val, firing_labels[val_indices])
sil_test_firing = silhouette_score(zloc_test, firing_labels[test_indices])
sil_train_batch = silhouette_score(zloc_tr, group_sample_labels[train_indices])
sil_val_batch = silhouette_score(zloc_val, group_sample_labels[val_indices])
sil_test_batch = silhouette_score(zloc_test, group_sample_labels[test_indices])  # Uncomment if needed
superprint(f'Silhouette training (firing): {sil_train_firing:.4f}')
superprint(f'Silhouette validation (firing): {sil_val_firing:.4f}')
superprint(f'Silhouette validation (firing): {sil_test_firing:.4f}')
superprint(f'Silhouette training (batch): {sil_train_batch:.4f}')
superprint(f'Silhouette validation (batch): {sil_val_batch:.4f}')

# 2. train linear SVM classifier on zloc_tr and evaluate on zloc_val
ba,prec,rec,f1 = None, None, None , None
# ba,prec,rec,f1 = evaluate_latent_svm(zloc_tr, firing_labels[train_indices], zloc_val, firing_labels[val_indices])
# superprint(f'Balanced accuracy: {ba:.4f}')
# superprint(f'Precision: {prec:.4f}')
# superprint(f'Recall: {rec:.4f}')
# superprint(f'F1: {f1:.4f}')

## 3. ARI for HDBSCAN

clust_train, ari_train = get_hdbscan_ari(zloc_tr, firing_labels[train_indices])
clust_val, ari_val = get_hdbscan_ari(zloc_val, firing_labels[val_indices])
clust_test, ari_test = get_hdbscan_ari(zloc_test, firing_labels[test_indices])
superprint(f'ARI train: {ari_train:.4f}')
superprint(f'ARI val: {ari_val:.4f}')
superprint(f'ARI test: {ari_test:.4f}')

## 4. Local entropy (firing dynamics)

med_ent_firing_tr = compute_local_entropy(zloc_tr, firing_labels[train_indices], k=100)
med_ent_batch_tr = compute_local_entropy(zloc_tr, group_sample_labels[train_indices], k=100)
med_ent_firing_val = compute_local_entropy(zloc_val, firing_labels[val_indices], k=20)
med_ent_batch_val = compute_local_entropy(zloc_val, group_sample_labels[val_indices], k=20)
med_ent_firing_test = compute_local_entropy(zloc_test, firing_labels[test_indices], k=20)
med_ent_batch_test = compute_local_entropy(zloc_test, group_sample_labels[test_indices], k=20)

superprint(f'Median local entropy train (firing): {med_ent_firing_tr:.4f} bits')
superprint(f'Median local entropy train (batch): {med_ent_batch_tr:.4f} bits')
superprint(f'Median local entropy val (firing): {med_ent_firing_val:.4f} bits')
superprint(f'Median local entropy val (batch): {med_ent_batch_val:.4f} bits')
superprint(f'Median local entropy test (firing): {med_ent_firing_test:.4f} bits')
superprint(f'Median local entropy test (batch): {med_ent_batch_test:.4f} bits')

## 5. kBET

kbet_train = kBET(zloc_tr, group_sample_labels[train_indices])
kbet_val = kBET(zloc_val, group_sample_labels[val_indices])
kbet_test = kBET(zloc_test, group_sample_labels[test_indices])
superprint(f"Rejection rate kBET (train): {kbet_train:.3f}")
superprint(f"Rejection rate kBET (val): {kbet_val:.3f}")
superprint(f"Rejection rate kBET (test): {kbet_test:.3f}")

#########################################################################################################
# Save results
#########################################################################################################

superprint('Storing summary...')

# Create a dataframe with the required information, including test metrics
results_df = pd.DataFrame({
    'seed': [seed],
    'fluo_noise': [fluo_noise],
    'model': ['BFA'],
    'model_type': ['Unsupervised'],
    'mask': [False],
    'out_var': [None],
    'norm': [None],
    'beta': [1.0],
    'lr': [lr],
    'latent_dim': [latent_dim],
    'target_dist': ['Normal'],
    'mae_x_val': [mae_x_val],
    'mse_x_val': [mse_x_val],
    'mae_x_test': [mae_x_test],
    'mse_x_test': [mse_x_test],
    'ba': [ba],
    'precision': [prec],
    'recall': [rec],
    'f1': [f1],
    'ari_train': [ari_train],
    'ari_val': [ari_val],
    'ari_test': [ari_test],
    'sil_train': [sil_train_firing],
    'sil_val': [sil_val_firing],
    'sil_test': [sil_test_firing],
    'sil_train_batch': [sil_train_batch],
    'sil_val_batch': [sil_val_batch],
    'sil_test_batch': [sil_test_batch],  # If you want to add test batch, use sil_val_batch or compute sil_test_batch
    'med_ent_firing_train': [med_ent_firing_tr],
    'med_ent_batch_train': [med_ent_batch_tr],
    'med_ent_firing_val': [med_ent_firing_val],
    'med_ent_batch_val': [med_ent_batch_val],
    'med_ent_firing_test': [med_ent_firing_test],
    'med_ent_batch_test': [med_ent_batch_test],
    'kbet_train': [kbet_train],
    'kbet_val': [kbet_val],
    'kbet_test': [kbet_test],
    'firing_acc': [firing_acc],
    'batch_acc': [batch_acc]
})

#%%

# Append the dataframe to a text file
results_file = f'{outdir}/results_summary_effa_{domain}_test_group_23_no_FS.txt'
header = not os.path.exists(results_file)
results_df.to_csv(results_file, mode='a', header=header, index=False, sep='\t')
