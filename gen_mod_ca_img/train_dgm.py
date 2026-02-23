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

# PyTorch
import torch
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Our modules
from ca_sn_gen_models.utils import superprint, running_in_ipython
from ca_sn_gen_models.evaluation import kBET
from ca_sn_gen_models.evaluation import get_hdbscan_ari
from ca_sn_gen_models.evaluation import evaluate_latent_svm
from ca_sn_gen_models.evaluation import compute_local_entropy

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superprint(f"Using device: {device}")

###############################################################################################
# arguments
###############################################################################################
#%%

if running_in_ipython():
    seed = 1
    lr = 0.0001
    num_epochs = 1000
    latent_dim = 8
    fluo_noise = 1.0
    batch_size = 1000
    vae = 'FixedVarMlpVAE'
    outdir = '/pool01/projects/abante_lab/snDGM/icassp2026/results/'

    num_ind_pts = None
    target_dist = 'Normal'
    if vae == 'SupMlpGpVAE':
        num_ind_pts = 1500
        target_dist = 'GP'

else:
    # Create the parser
    parser = argparse.ArgumentParser(description="FixedVarMlpVAE training script")

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
        default=100,
        help="Dimension of latent space (default: 100)"
    )

    parser.add_argument(
        "-e",
        "--num_epochs", 
        type=int, 
        required=False,
        default=2000,
        help="Dimension of latent space (default: 5000)"
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
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )

    parser.add_argument(
        "-b",
        "--batch_size", 
        type=int, 
        required=False, 
        default=256,
        help="Batch size for training (default: 256)"
    )

    parser.add_argument(
        '--outdir', 
        type=str, 
        required=False,
        default='./output', 
        help='Folder to save output files'
    )

    parser.add_argument(
        '--num_ind_pts',
        type=int,
        required=False,
        default=1500,
        help='Number of inducing points for GP models (default: 1500)'
    )

    parser.add_argument(
        '--target_dist',
        type=str,
        required=False,
        default='GP',
        help='Target distribution for GP models (default: GP)'
    )

    # valid model
    valid_models = [
        'FixedVarMlpVAE',
        'LearnedVarMlpVAE',
        'FixedVarSupMlpVAE',
        'LearnedVarSupMlpVAE',
        'FixedVarSupMlpDenVAE',
        'LearnedVarSupMlpDenVAE',
        'SupMlpGpVAE'
    ]

    parser.add_argument(
        '--vae', 
        type=str, 
        required=False,
        default='FixedVarMlpVAE',
        choices=valid_models,
        help='Model to use (default: FixedVarMlpVAE). Options: ' + ', '.join(valid_models)
    )

    # Parse the arguments
    args = parser.parse_args()

    # Access the arguments
    vae = args.vae
    lr = args.rate
    seed = args.seed
    outdir = args.outdir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    fluo_noise = args.fluo_noise
    latent_dim = args.latent_dim

    num_ind_pts = None
    target_dist = 'Normal'
    if vae == 'SupMlpGpVAE':
        num_ind_pts = args.num_ind_pts
        target_dist = args.target_dist

###############################################################################################
# import model
###############################################################################################
#%%

uses_gp = False

if vae == 'FixedVarMlpVAE':
    from ca_sn_gen_models.models import FixedVarMlpVAE as vae_model
    model_type = 'Unsupervised'
    out_var = 'Fixed'
    mask = False
elif vae == 'LearnedVarMlpVAE':
    from ca_sn_gen_models.models import LearnedVarMlpVAE as vae_model
    model_type = 'Unsupervised'
    out_var = 'Learned'
    mask = False
elif vae == 'FixedVarSupMlpVAE':
    from ca_sn_gen_models.models import FixedVarSupMlpVAE as vae_model
    model_type = 'Supervised'
    out_var = 'Fixed'
    mask = False
elif vae == 'LearnedVarSupMlpVAE':
    from ca_sn_gen_models.models import LearnedVarSupMlpVAE as vae_model
    model_type = 'Supervised'
    out_var = 'Learned'
    mask = False
elif vae == 'FixedVarSupMlpDenVAE':
    from ca_sn_gen_models.models import FixedVarSupMlpDenVAE as vae_model
    model_type = 'Supervised'
    out_var = 'Fixed'
    mask = True
elif vae == 'LearnedVarSupMlpDenVAE':
    from ca_sn_gen_models.models import LearnedVarSupMlpDenVAE as vae_model
    model_type = 'Supervised'
    out_var = 'Learned'
    mask = True
elif vae == 'SupMlpGpVAE':
    from ca_sn_gen_models.models import SupMlpGpVAE as vae_model
    model_type = 'Supervised'
    out_var = 'GP'
    mask = False
    uses_gp = True
else:
    raise ValueError(f"Model {vae} not recognized. Choose from {valid_models}.")

#########################################################################################################
# Read in
#########################################################################################################

## Simulated calcium traces

superprint('Reading in data...')

# included groups
group_set = [2, 3]
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

# concatenate data
x = torch.cat(data_list, dim=0)

# subsample data
x = x[:,:10000]

# compute average signal for all xs
x_mean = x.mean(dim=1, keepdim=True)

# plot three examples from each firing type firing_labels
fig, axs = plt.subplots(len(np.unique(firing_labels)), 1, figsize=(10, 8))
for i, firing_type in enumerate(np.unique(firing_labels)):
    firing_indices = np.where(firing_labels == firing_type)[0]
    random_indices = np.random.choice(firing_indices, 3, replace=False)
    for j, idx in enumerate(random_indices):
        axs[i].plot(x[idx], alpha=0.5)
        axs[i].set_title(f"Firing Type {firing_type} Example {idx}")
        axs[i].set_xlabel("Frequency")
        axs[i].set_ylabel("Amplitude")
plt.tight_layout()
plt.show()

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


# #########################################################################################################
# SPLIT DATA
#########################################################################################################

superprint('Splitting data...')

data = x

# split and create data loader
train_val_indices, test_indices = train_test_split(np.arange(len(data)), test_size=0.1, random_state=seed)

train_indices, val_indices = train_test_split(train_val_indices, test_size=0.11, random_state=seed)

# split data
train_data = data[train_indices]
val_data = data[val_indices]
test_data = data[test_indices]

if model_type == 'Supervised':
    superprint('Using supervised model.')

    # split labels
    train_labels = oh_encoded_labels[train_indices]
    val_labels = oh_encoded_labels[val_indices]
    test_labels = oh_encoded_labels[test_indices]

    # create datasets
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)   
else:
    superprint('Using unsupervised model.')

    # create datasets
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    test_dataset = TensorDataset(test_data)             

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#########################################################################################################
# TRAIN
#########################################################################################################
#%%
# model path
if uses_gp:
    model_path = f'{outdir}/models/{vae}_seed_{seed}_k_{latent_dim}_m_{num_ind_pts}_sigma_{fluo_noise}_test_group_5.pt'
else:
    model_path = f'{outdir}/models/{vae}_seed_{seed}_latent_{latent_dim}_fluo_{fluo_noise}_test_group_5.pt'

# init model
superprint(f'Initializing model {vae} with latent dimension {latent_dim} and noise level {fluo_noise}...')
if model_type == 'Supervised':
    
    # Initialize model with num_classes
    if uses_gp:
        model = vae_model(data.shape[1], latent_dim, num_classes, M=num_ind_pts, device=device)
    else:
        model = vae_model(data.shape[1], latent_dim, num_classes, device=device)

else:
    
    # Initialize model
    model = vae_model(data.shape[1], latent_dim, device=device)

if not os.path.exists(model_path):
    
    superprint(f'Model does not exist at {model_path}. Starting training...')

    # Start the timer
    start_time = time.time()

    # clear cuda memory
    torch.cuda.empty_cache()

    # Clear Pyro parameters
    pyro.clear_param_store()

    # Train the model
    loss_tr,loss_val = model.train_model(train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=50, min_delta=1e-2)

    # Monitor peak RAM usage
    process = psutil.Process()

    # Get peak memory usage in MB
    peak_memory = process.memory_info().peak_wset / (1024 ** 2) if hasattr(process.memory_info(), 'peak_wset') else process.memory_info().rss / (1024 ** 2)
    superprint(f"Peak RAM usage: {peak_memory:.2f} MB")

    # stop timer
    end_time = time.time()
    training_time = end_time - start_time
    superprint(f'Training time: {training_time:.2f} seconds')

    # Save the model
    torch.save(model.state_dict(), model_path)
    superprint(f'Model saved to {model_path}')

    # plot training and validation loss
    fig, ax1 = plt.subplots()

    # Plot train loss on the first y-axis
    ax1.plot(range(len(loss_tr)), loss_tr, label='Train Loss', color='blue')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(range(len(loss_val)), loss_val, label='Validation Loss', color='red')
    ax2.set_ylabel('Validation Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add a title and show the plot
    plt.title('Train and Validation Loss')
    fig.tight_layout()
    plt.show()

else:
    
    superprint(f'Model already exists at {model_path}. Loading model...')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    training_time = None
    peak_memory = None

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

# concatenate data
x0 = torch.cat(data_list, dim=0)

# subsample data
x0 = x0[:,:10000]

# compute average signal for all xs
x0_mean = x0.mean(dim=1, keepdim=True)

# plot three examples from each firing type firing_labels
fig, axs = plt.subplots(len(np.unique(firing_labels)), 1, figsize=(10, 8))
for i, firing_type in enumerate(np.unique(firing_labels)):
    firing_indices = np.where(firing_labels == firing_type)[0]
    random_indices = np.random.choice(firing_indices, 3, replace=False)
    for j, idx in enumerate(random_indices):
        axs[i].plot(x0[idx], alpha=0.5)
        axs[i].set_title(f"Firing Type {firing_type} Example {idx}")
        axs[i].set_xlabel("Frequency")
        axs[i].set_ylabel("Amplitude")
plt.tight_layout()
plt.show()

x0 = x0[meta_df.index]

# split into train and validation
x0_train = x0[train_indices]
x0_val = x0[val_indices]
x0_test = x0[test_indices]

#########################################################################################################
# INFERENCE
#########################################################################################################

superprint('Getting posterior estimates...')

# Set model to evaluation mode
model.eval()

# get posterior value of Z given X and Y for training data
zloc_tr = []
xhat_tr = []

for batch in train_loader:
    
    if model_type == 'Supervised':
        
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
    else:

        x_batch = batch[0].to(device=device)
        
        with torch.no_grad():
            
            zloc_tr_batch, _ = model.encode(x_batch)
            zloc_tr.append(zloc_tr_batch.detach().cpu())
            
            if out_var == 'Fixed':
                xhat_tr_batch = model.decode(zloc_tr_batch)
            else:
                xhat_tr_batch,_ = model.decode(zloc_tr_batch)
            
            xhat_tr.append(xhat_tr_batch.detach().cpu())

# concatenate batches
zloc_tr = torch.cat(zloc_tr, dim=0)
xhat_tr = torch.cat(xhat_tr, dim=0)

# get posterior value of Z given X and Y for validation data
zloc_val = []
zscl_val = []
xhat_val = []
for batch in val_loader:

    if model_type == 'Supervised':
        
        x_batch = batch[0].to(device=device)
        y_batch = batch[1].to(device=device)
        
        with torch.no_grad():
            
            if mask:
                # For masked models, we need to pass the mask as well
                zloc_val_batch, _ = model.encode(x_batch, y_batch, torch.ones_like(x_batch, dtype=torch.float32))
            else:
                # For non-masked models, we can just pass x_batch and y_batch
                zloc_val_batch, _ = model.encode(x_batch, y_batch)
            
            zloc_val.append(zloc_val_batch.detach().cpu())
            
            if out_var == 'Fixed':
                xhat_val_batch = model.decode(zloc_val_batch, y_batch)
            else:  # Learned
                xhat_val_batch,_ = model.decode(zloc_val_batch, y_batch)
            
            xhat_val.append(xhat_val_batch.detach().cpu())
    else:

        x_batch = batch[0].to(device=device)
        
        with torch.no_grad():
            
            zloc_val_batch, _ = model.encode(x_batch)
            zloc_val.append(zloc_val_batch.detach().cpu())
            
            if out_var == 'Fixed':
                xhat_val_batch = model.decode(zloc_val_batch)
            else:
                xhat_val_batch,_ = model.decode(zloc_val_batch)
            
            xhat_val.append(xhat_val_batch.detach().cpu())

# concatenate batches
zloc_val = torch.cat(zloc_val, dim=0)
xhat_val = torch.cat(xhat_val, dim=0)

zloc_test = []
zscl_test = []
xhat_test = []

for batch in test_loader:

    if model_type == 'Supervised':
        
        x_batch = batch[0].to(device=device)
        y_batch = batch[1].to(device=device)
        
        with torch.no_grad():
            
            if mask:
                # For masked models, we need to pass the mask as well
                zloc_test_batch, _ = model.encode(x_batch, y_batch, torch.ones_like(x_batch, dtype=torch.float32))
            else:
                # For non-masked models, we can just pass x_batch and y_batch
                zloc_test_batch, _ = model.encode(x_batch, y_batch)
            
            zloc_test.append(zloc_test_batch.detach().cpu())
            
            if out_var == 'Fixed':
                xhat_test_batch = model.decode(zloc_test_batch, y_batch)
            else:  # Learned
                xhat_test_batch,_ = model.decode(zloc_test_batch, y_batch)
            
            xhat_test.append(xhat_test_batch.detach().cpu())
    else:

        x_batch = batch[0].to(device=device)
        
        with torch.no_grad():
            
            zloc_test_batch, _ = model.encode(x_batch)
            zloc_test.append(zloc_test_batch.detach().cpu())
            
            if out_var == 'Fixed':
                xhat_test_batch = model.decode(zloc_test_batch)
            else:
                xhat_test_batch,_ = model.decode(zloc_test_batch)
            
            xhat_test.append(xhat_test_batch.detach().cpu())

zloc_test = torch.cat(zloc_test, dim=0)
xhat_test = torch.cat(xhat_test, dim=0)

# #########################################################################################################
# # Check correlation of each latent dimension with firing type of each neuron

# Obtain int labels for FS, RS, LTS
firing_type_mapping = {'FS': 0, 'RS': 1, 'LTS': 2, 'CH': 3}
firing_type_int = np.array([firing_type_mapping[ft] for ft in firing_labels])

# Train a classifier to predict firing type from latent variables
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Initialize classifier
clf = RandomForestClassifier(n_estimators=100, random_state=seed)
# Train on training set
clf.fit(zloc_tr.numpy(), firing_type_int[train_indices])
# Predict on test set
y_pred = clf.predict(zloc_test.numpy())
# Print classification report
print(classification_report(firing_type_int[test_indices], y_pred, target_names=['RS', 'LTS']))
firing_acc = (y_pred == firing_type_int[test_indices]).mean()

# Train a classifier to predict batch label from latent variables
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# Initialize classifier
clf = RandomForestClassifier(n_estimators=100, random_state=seed)
# Train on training set
clf.fit(zloc_tr.numpy(), encoded_labels[train_indices].numpy())
# Predict on test set
y_pred = clf.predict(zloc_test.numpy())
# Print classification report
print(classification_report(encoded_labels[test_indices].numpy(), y_pred, target_names=encoded_labels.unique().numpy().astype(str)))
batch_acc = (y_pred == encoded_labels[test_indices].numpy()).mean()

#########################################################################################################
# EMBEDDING
#########################################################################################################

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
    'MeanFiring': x_mean[train_indices].squeeze(1).cpu().numpy()
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
plt.figure(figsize=(8, 5))
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

##############################################################################################################
# Reconstruction of X
##############################################################################################################

superprint('Evaluating reconstruction...')

# plot a few examples of the data
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
random_indices = np.random.choice(len(val_data), 4, replace=False)
for i, idx in enumerate(random_indices):
    axs[i].plot(x0_val[idx], label='GT', color='green')
    axs[i].plot(xhat_val[idx], label='Reconstructed FA', color='red', alpha=0.5)
    axs[i].set_title(f'Sample {idx}')
    axs[i].set_xlabel('Frequency')
    axs[i].set_ylabel('Amplitude')
    axs[i].set_ylim(-1, 1)
plt.legend()
plt.tight_layout()
plt.show()

# compute reconstruction error with noiseless data
mae_x_val = torch.nn.functional.l1_loss(x0_val, xhat_val, reduction='mean').item()
mse_x_val = torch.nn.functional.mse_loss(x0_val, xhat_val, reduction='mean').item()

# Compute reconstruction error of test set
mae_x_test = torch.nn.functional.l1_loss(x0_test, xhat_test, reduction='mean').item()
mse_x_test = torch.nn.functional.mse_loss(x0_test, xhat_test, reduction='mean').item()

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
sil_test_batch = silhouette_score(zloc_test, group_sample_labels[test_indices])

superprint(f'Silhouette training (firing): {sil_train_firing:.4f}')
superprint(f'Silhouette validation (firing): {sil_val_firing:.4f}')
superprint(f'Silhouette testing (firing): {sil_test_firing:.4f}')
superprint(f'Silhouette training (batch): {sil_train_batch:.4f}')
superprint(f'Silhouette validation (batch): {sil_val_batch:.4f}')
superprint(f'Silhouette testing (batch): {sil_test_batch:.4f}')

# 2. train linear SVM classifier on zloc_tr and evaluate on zloc_val
ba, prec, rec, f1 = None, None, None, None
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
med_ent_batch_test = compute_local_entropy(zloc_test, group_sample_labels[test_indices], k=20)
med_ent_firing_test = compute_local_entropy(zloc_test, firing_labels[test_indices], k=20)       

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

# Create a dataframe with the required information
results_df = pd.DataFrame({
    'seed': [seed],
    'fluo_noise': [fluo_noise],
    'model': [model.__class__.__name__],
    'model_type': [model_type],
    'mask': [mask],
    'out_var': [out_var],
    'norm': [None],
    'beta': [1.0],
    'lr': [lr],
    'latent_dim': [latent_dim],
    'num_ind_pts': [num_ind_pts],
    'target_dist': [target_dist],
    'training_time': [training_time],
    'peak_memory': [peak_memory],
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
    'sil_test_batch': [sil_test_batch],
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
    'batch_acc': [batch_acc],
})

#%%
# Append the dataframe to a text file
results_file = f'{outdir}/results_summary_model_{model.__class__.__name__}_test_group_23_no_FS.txt'
header = not os.path.exists(results_file)
results_df.to_csv(results_file, mode='a', header=header, index=False, sep='\t')
