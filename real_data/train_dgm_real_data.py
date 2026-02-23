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
from ca_sn_gen_models.utils import superprint
from ca_sn_gen_models.evaluation import kBET
from ca_sn_gen_models.evaluation import compute_local_entropy

def running_in_ipython():
    try:
        get_ipython   # type: ignore
        return True
    except NameError:
        return False
    
# Detect device
# device ='cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
superprint(f"Using device: {device}")

###############################################################################################
# arguments
###############################################################################################

if running_in_ipython():
    # Create the parser
    parser = argparse.ArgumentParser(description="Training script for real data (MLCB 2025).")

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
        default=5000,
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
        default=20000,
        help="Batch size for training (default: 20000)"
    )

    parser.add_argument(
        '--outdir', 
        type=str, 
        required=False,
        default='./output', 
        help='Folder to save output files'
    )

    # valid model
    valid_models = [
        'FixedVarMlpVAE',
        'LearnedVarMlpVAE',
        'FixedVarSupMlpVAE',
        'LearnedVarSupMlpVAE',
        'FixedVarSupMlpDenVAE',
        'LearnedVarSupMlpDenVAE'
    ]

    parser.add_argument(
        '--vae', 
        type=str, 
        required=False,
        default='FixedVarSupMlpVAE',
        choices=valid_models,
        help='Model to use (default: FixedVarSupMlpVAE). Options: ' + ', '.join(valid_models)
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
    latent_dim = args.latent_dim

else:

    # Comment out the following lines if you want to run the script without command line arguments
    seed = 1                    # RNG seed
    roll = False                # Whether to use rolling window in each batch
    lr = 0.0001                 # Learning rate
    scl = 0.5                # Scale for the output model variance
    latent_dim = 4              # Dimension of latent space
    num_epochs = 1000           # Number of epochs to train
    batch_size = 1000           # Batch size for training
    vae = 'FixedVarSupMlpVAE'      # Model to use
    outdir = '/pool01/projects/abante_lab/snDGM/mlcb_2025/results/real_data'

###############################################################################################
# import model
###############################################################################################

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
else:
    raise ValueError(f"Model {vae} not recognized. Choose from {valid_models}.")

#########################################################################################################
# Read in
#########################################################################################################

## Simulated calcium traces

superprint('Reading in data...')

# directory
data_dir = '/pool01/data/private/soriano_lab/processed/mireia_spring_2025/'

# metadata
meta_df = pd.read_csv(f'{data_dir}/metadata.txt', sep='\t')
meta_df['SampleID'] = meta_df.index

# traces
data_list = []
sampleid_list = []
for idx, row in meta_df.iterrows():
    
    # get metadata
    div = row['div']
    rep = row['replicate']
    org = row['organism']
    
    # path 
    data_path_sims = f'{data_dir}/div_{div}/{org.upper()}/div{div}_{org}_rep{rep}.csv'
    
    # read in data
    data_sims = pd.read_csv(data_path_sims)
    data_sims = data_sims.drop(columns=['time (s)'])  # drop time column
    
    # create metadata table
    sampleid_list.append(torch.tensor([row['SampleID']] * data_sims.shape[1], dtype=torch.long))

    # append to lists
    data_list.append(torch.tensor(data_sims.T.values, dtype=torch.float32))

# get maximum length of traces and pad shorter traces with NaN
max_length = max([data.shape[1] for data in data_list])

# normalize between -1 and 1 and pad data with NaN
for i in range(len(data_list)):
    
    # normalize data
    data_list[i] = (data_list[i] - data_list[i].mean(dim=1, keepdim=True)) / (10 * data_list[i].std(dim=1, keepdim=True))
    
    # pad data with NaN
    if data_list[i].shape[1] > max_length:
        data_list[i] = data_list[i][:, :max_length]
    elif data_list[i].shape[1] < max_length:
        # Create padding with NaN values
        padding = torch.full((data_list[i].shape[0], max_length - data_list[i].shape[1]), float('nan'))
        # Concatenate the original data with the padding
        data_list[i] = torch.cat((data_list[i], padding), dim=1)

# concatenate data
x = torch.cat(data_list, dim=0)
y = torch.cat(sampleid_list, dim=0)
superprint(f'Loaded {len(x)} traces with {x.shape[1]} time points each.')
assert x.shape[0] == y.shape[0], "Number of traces and labels must match."

# Compute min and max excluding NaNs
x_min = torch.from_numpy(np.nanmin(x.numpy(), axis=1, keepdims=True))
x_max = torch.from_numpy(np.nanmax(x.numpy(), axis=1, keepdims=True))
superprint(f'Range of values (excluding NaN): min={x_min.min().item()}, max={x_max.max().item()}')

# compute average signal for all xs
x_mean = torch.nanmean(x, dim=1, keepdim=True)

# Plot 4 examples for each unique label in y
superprint('Plotting 4 examples for each label in y...')
unique_labels = torch.unique(y)
num_examples = 4
fig, axs = plt.subplots(len(unique_labels), num_examples, figsize=(num_examples * 4, len(unique_labels) * 2.5), squeeze=False)
for i, label in enumerate(unique_labels):
    label_indices = (y == label).nonzero(as_tuple=True)[0].cpu().numpy()
    chosen_indices = np.random.choice(label_indices, num_examples, replace=True)
    for j, idx in enumerate(chosen_indices):
        axs[i, j].plot(x[idx].cpu().numpy(), color='blue')
        axs[i, j].set_title(f'Label {int(label.item())} - Example {j+1}')
        axs[i, j].set_xlabel('Time')
        axs[i, j].set_ylabel('Signal')
plt.tight_layout()
plt.show()

# Encode group_sample_labels using LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y)
encoded_labels = torch.tensor(encoded_labels, dtype=torch.long)

# Convert to one-hot encoding
num_classes = len(np.unique(encoded_labels))
yoh = torch.nn.functional.one_hot(encoded_labels,num_classes=num_classes)

#########################################################################################################
# SPLIT DATA
#########################################################################################################

superprint('Splitting data...')

# split and create data loader
train_val_indices, test_indices = train_test_split(np.arange(len(x)), test_size=0.1, random_state=seed)

train_indices, val_indices = train_test_split(train_val_indices, test_size=0.11, random_state=seed)

# split data
train_data = x[train_indices]
val_data = x[val_indices]
test_data = x[test_indices]

# split labels
train_labels = yoh[train_indices]
val_labels = yoh[val_indices]
test_labels = yoh[test_indices]

# # split and create data loader
# train_indices, val_indices = train_test_split(np.arange(x.shape[0]), test_size=0.2, random_state=seed)

# # split data
# train_data = x[train_indices]
# val_data = x[val_indices]

# # split labels
# train_labels = yoh[train_indices]
# val_labels = yoh[val_indices]

if model_type == 'Supervised':

    # # create datasets
    # train_dataset = TensorDataset(train_data, train_labels)
    # val_dataset = TensorDataset(val_data, val_labels)

    # # create data loaders
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # create datasets
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

else:

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

# model path
model_suff = f'{vae}_seed_{seed}_latent_{latent_dim}_scl_{scl}'
model_suff += '_roll' if roll else ''
model_path = f'{outdir}/{model_suff}_test.pt'

# init model
superprint(f'Initializing model {vae} with latent dimension {latent_dim}...')
if model_type == 'Supervised':
    
    # Initialize model with num_classes
    model = vae_model(x.shape[1], latent_dim, num_classes, scl=scl, device=device)

else:
    
    # Initialize model
    model = vae_model(x.shape[1], latent_dim, device=device)

if not os.path.exists(model_path):
    
    superprint(f'Model does not exist at {model_path}. Starting training...')

    # Start the timer
    start_time = time.time()

    # clear cuda memory
    torch.cuda.empty_cache()

    # Clear Pyro parameters
    pyro.clear_param_store()

    # Train the model
    loss_tr,loss_val = model.train_model(train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=50, min_delta=1e-2, roll=True)

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



# get posterior value of Z given X and Y for validation data
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

# concatenate batches
zloc_test = torch.cat(zloc_test, dim=0)
xhat_test = torch.cat(xhat_test, dim=0)

# concatenate Zloc for training and validation data
zloc = torch.cat((zloc_tr, zloc_val, zloc_test), dim=0)

# order sampleid labels
y_ord = torch.cat((y[train_indices], y[val_indices], y[test_indices]), dim=0)

#########################################################################################################
# EMBEDDING
#########################################################################################################

# Perform UMAP dimensionality reduction
reducer = umap.UMAP(n_components=2)
Z_umap = reducer.fit_transform(zloc)

# create meta table from meta_df using y_ord as indexes
meta_aug_df = meta_df.loc[y_ord.numpy()]
meta_aug_df.reset_index(drop=True, inplace=True)

# # Create a DataFrame for easier plotting with seaborn
# umap_df = pd.DataFrame({
#     'UMAP1': Z_umap[:, 0],
#     'UMAP2': Z_umap[:, 1],
#     'Mean': x_mean[y_ord].numpy().flatten()
# })

# # merge with metadata
# umap_df = umap_df.merge(meta_aug_df, left_index=True, right_index=True)

# # Plot using seaborn for Group and Sample, and save as PDF in outdir
# plt.figure(figsize=(8, 5))
# scatter = sns.scatterplot(
#     data=umap_df,
#     x='UMAP1',
#     y='UMAP2',
#     hue='SampleID',
#     style='organism',
#     palette='tab20',
#     s=50
# )
# # Rasterize the scatter points to reduce PDF size
# for coll in scatter.collections:
#     coll.set_rasterized(True)
# plt.tight_layout()
# plt.savefig(f"{outdir}/umap_scatter_sampleid_org_{model_suff}.png")
# plt.show()
# plt.close()

# plt.figure(figsize=(8, 5))
# scatter = sns.scatterplot(
#     data=umap_df,
#     x='UMAP1',
#     y='UMAP2',
#     hue='date',
#     style='organism',
#     palette='tab20',
#     s=50
# )
# plt.tight_layout()
# plt.show()

# # Plot using seaborn for Firing Type
# plt.figure(figsize=(8, 5))
# scatter = plt.scatter(
#     umap_df['UMAP1'],
#     umap_df['UMAP2'],
#     c=umap_df['Mean'],
#     cmap='viridis',
#     s=5,
#     alpha=0.8
# )
# plt.colorbar(scatter, label='Mean Firing')
# plt.title("UMAP of Latent Variables Z (Colored by Mean Firing Rate)")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.tight_layout()
# plt.show()

# # Define a single palette for date, common across plots
# unique_dates = sorted(umap_df['date'].unique())
# date_palette = dict(zip(unique_dates, sns.color_palette('tab10', len(unique_dates))))

# # Plot UMAPs in a 2x2 grid: rows=div, columns=organism, hue=SampleID
# divs = sorted(umap_df['div'].unique())
# orgs = sorted(umap_df['organism'].unique())
# fig, axs = plt.subplots(len(divs), len(orgs), figsize=(6 * len(orgs), 4 * len(divs)), squeeze=False)

# # Get global axis limits
# xlim = (umap_df['UMAP1'].min()-1, umap_df['UMAP1'].max()+1)
# ylim = (umap_df['UMAP2'].min()-1, umap_df['UMAP2'].max()+1)

# for i, div in enumerate(divs):
#     for j, org in enumerate(orgs):
#         ax = axs[i, j]
#         subset = umap_df[(umap_df['div'] == div) & (umap_df['organism'] == org)]
#         if not subset.empty:
#             sns.scatterplot(
#                 data=subset,
#                 x='UMAP1',
#                 y='UMAP2',
#                 hue='date',
#                 palette=date_palette,
#                 s=50,
#                 ax=ax,
#                 legend=True
#             )
#         ax.set_title(f"div={div}, organism={org}")
#         ax.set_xlabel("UMAP1")
#         ax.set_ylabel("UMAP2")
#         ax.set_xlim(xlim)
#         ax.set_ylim(ylim)
# plt.tight_layout()
# plt.savefig(f"{outdir}/umap_scatter_grid_div_org_{model_suff}.pdf")
# plt.show()
# plt.close()

##############################################################################################################
# Reconstruction of X
##############################################################################################################
superprint('Evaluating reconstruction...')

# get validation data
x_val = x[val_indices]
x_test = x[test_indices]

# plot three examples for each distinct value of y in the validation set, using same y axis throughout
unique_val_labels = torch.unique(y[val_indices])
num_examples = 3
num_labels = len(unique_val_labels)
fig, axs = plt.subplots(num_labels, num_examples, figsize=(5 * num_examples, 2.5 * num_labels), squeeze=False)

# Compute global y-limits (ignore NaNs)
all_vals = torch.cat([x_val.flatten(), xhat_val.flatten()])
all_vals = all_vals[~torch.isnan(all_vals)]
ymin, ymax = all_vals.min().item(), all_vals.max().item()

for i, label in enumerate(unique_val_labels):
    # find indices in val_indices with this label
    label_indices = (y[val_indices] == label).nonzero(as_tuple=True)[0].cpu().numpy()
    # pick up to 3 examples (random if more than 3, else with replacement)
    chosen_indices = np.random.choice(label_indices, num_examples, replace=len(label_indices) < num_examples)
    for j, idx in enumerate(chosen_indices):
        axs[i, j].plot(x_val.cpu()[idx], label='GT', color='green')
        axs[i, j].plot(xhat_val.cpu()[idx], label='Reconstructed', color='red', alpha=0.5)
        axs[i, j].set_title(f'Sample {int(label.item())} - Neuron {idx}')
        axs[i, j].set_xlabel('Time')
        axs[i, j].set_ylabel('Signal')
        axs[i, j].set_ylim([ymin, ymax])
        axs[i, j].legend()
plt.tight_layout()
plt.savefig(f"{outdir}/reconstruction_{model_suff}.pdf")
plt.show()

# Compute MAE ignoring NaNs
mae_x_val = torch.abs(x_val - xhat_val)
mae_x_val = mae_x_val[~torch.isnan(mae_x_val)]
mae_x_val = mae_x_val.mean().item()

# Compute MAE ignoring NaNs
mae_x_test = torch.abs(x_test - xhat_test)
mae_x_test = mae_x_test[~torch.isnan(mae_x_test)]
mae_x_test = mae_x_test.mean().item()
###############################################################################################
# Evaluation of the latent representation
###############################################################################################

superprint('Evaluating latent representation...')

## 1. silhouette scores

sil_train_batch = silhouette_score(zloc_tr, y[train_indices])
sil_val_batch = silhouette_score(zloc_val, y[val_indices])
sil_test_batch = silhouette_score(zloc_test, y[test_indices])
superprint(f'Silhouette training (batch): {sil_train_batch:.4f}')
superprint(f'Silhouette validation (batch): {sil_val_batch:.4f}')

## 2. Local entropy (firing dynamics)

med_ent_batch_tr = compute_local_entropy(zloc_tr, y[train_indices], k=100)
med_ent_batch_val = compute_local_entropy(zloc_val, y[val_indices], k=20)
med_ent_batch_test = compute_local_entropy(zloc_test, y[test_indices], k=20)
superprint(f'Median local entropy train (batch): {med_ent_batch_tr:.4f} bits')
superprint(f'Median local entropy val (batch): {med_ent_batch_val:.4f} bits')

## 3. kBET

kbet_train = kBET(zloc_tr, y[train_indices].numpy())
kbet_val = kBET(zloc_val, y[val_indices].numpy())
kbet_test = kBET(zloc_test, y[test_indices].numpy())
superprint(f"Rejection rate kBET (train): {kbet_train:.3f}")
superprint(f"Rejection rate kBET (val): {kbet_val:.3f}")

###############################################################################################
# Clustering in latent space
###############################################################################################
#%%
# Perform clustering in latent space using k-means
# from sklearn.cluster import KMeans

# superprint('Performing clustering in latent space...')
# kmeans = KMeans(n_clusters=4, random_state=seed)
# kmeans.fit(zloc)

# # Get cluster labels
# cluster_labels = kmeans.labels_

# # add clustering results to the UMAP DataFrame
# umap_df['Cluster'] = cluster_labels

# # Plot UMAP with cluster labels
# plt.figure(figsize=(8, 5))
# scatter = sns.scatterplot(
#     data=umap_df,
#     x='UMAP1',
#     y='UMAP2',
#     hue='Cluster',
#     palette='tab20',
#     s=10
# )
# plt.title('UMAP Embedding Colored by Cluster')
# plt.xlabel('UMAP1')
# plt.ylabel('UMAP2')
# plt.tight_layout()

# # plot violin plot of Mean per cluster
# plt.figure(figsize=(8, 5))
# sns.violinplot(
#     x='Cluster',
#     y='Mean',
#     data=umap_df,
#     hue='Cluster',
#     palette='tab20',
#     inner='quartile'
# )
# plt.title('Mean Firing Rate per Cluster')
# plt.xlabel('Cluster')
# plt.ylabel('Mean Firing Rate')
# plt.tight_layout()
 
# # plot heatmap of contribution of each SampleID to each cluster, normalized per row
# contrib = pd.crosstab(umap_df['SampleID'], umap_df['Cluster'])
# contrib_norm = contrib.div(contrib.sum(axis=1), axis=0)

# # Optional: Add row/column annotations (like pheatmap in R)
# # For row annotations, e.g., organism or div per SampleID
# row_annot = meta_df.set_index('SampleID').loc[contrib_norm.index][['organism', 'div']]

# # Convert categorical annotations to colors
# import matplotlib.patches as mpatches
# from matplotlib.colors import ListedColormap

# # Map organism to colors
# organism_palette = sns.color_palette("Set2", len(row_annot['organism'].unique()))
# organism_lut = dict(zip(row_annot['organism'].unique(), organism_palette))
# organism_colors = row_annot['organism'].map(organism_lut)

# # Map div to colors
# div_palette = sns.color_palette("Set1", len(row_annot['div'].unique()))
# div_lut = dict(zip(sorted(row_annot['div'].unique()), div_palette))
# div_colors = row_annot['div'].map(div_lut)

# # Combine row colors into a DataFrame for clustermap
# row_colors = pd.DataFrame({'organism': organism_colors, 'div': div_colors}, index=row_annot.index)

# # Plot heatmap with row colors using clustermap (best for annotation bars)
# g = sns.clustermap(
#     contrib_norm,
#     cmap='viridis',
#     annot=True,
#     fmt='.2f',
#     row_colors=row_colors,
#     linewidths=.5,
#     linecolor='black',
#     figsize=(10, 8),
#     cbar_kws={'label': 'Fraction'},
#     col_cluster=False, row_cluster=False
# )
# # Add legend for organism
# handles_org = [mpatches.Patch(color=organism_lut[name], label=name) for name in organism_lut]
# handles_div = [mpatches.Patch(color=div_lut[val], label=f'div {val}') for val in div_lut]
# plt.legend(handles=handles_org + handles_div, title='Annotations', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
# plt.title('SampleID Contribution to Clusters (Row-normalized)')
# plt.xlabel('Cluster')
# plt.ylabel('SampleID')
# plt.tight_layout()
# plt.show()

# # Plot 4 examples from each cluster in a single row
# superprint('Plotting 4 examples from each cluster...')
# num_examples_per_cluster = 4
# unique_clusters = sorted(umap_df['Cluster'].unique())
# fig, axs = plt.subplots(len(unique_clusters), num_examples_per_cluster, figsize=(5 * num_examples_per_cluster, 2.5 * len(unique_clusters)), squeeze=False)

# for i, cluster in enumerate(unique_clusters):
#     # Get indices of samples in this cluster
#     cluster_indices = umap_df[umap_df['Cluster'] == cluster].index.values
#     # Pick up to 4 examples (random if more than 4, else with replacement)
#     chosen_indices = np.random.choice(cluster_indices, num_examples_per_cluster, replace=len(cluster_indices) < num_examples_per_cluster)
#     for j, idx in enumerate(chosen_indices):
#         axs[i, j].plot(x[idx].cpu().numpy(), color='blue')
#         axs[i, j].set_title(f'Cluster {cluster} - Example {j+1} (SampleID {int(umap_df.loc[idx, "SampleID"])})')
#         axs[i, j].set_xlabel('Time')
#         axs[i, j].set_ylabel('Signal')
# plt.tight_layout()
# plt.show()

#########################################################################################################
# Save results
#########################################################################################################

superprint('Storing summary...')

# Create a dataframe with the required information
results_df = pd.DataFrame({
    'model': [model.__class__.__name__],
    'model_type': [model_type],
    'seed': [seed],
    'mask': [mask],
    'out_var': [out_var],
    'beta': [1.0],
    'lr': [lr],
    'latent_dim': [latent_dim],
    'mae_x': [mae_x_val],
    'mae_x_test': [mae_x_test],
    'sil_train_batch': [sil_train_batch],
    'sil_val_batch': [sil_val_batch],
    'sil_test_batch': [sil_test_batch],
    'med_ent_batch_train': [med_ent_batch_tr],
    'med_ent_batch_val': [med_ent_batch_val],
    'med_ent_batch_test': [med_ent_batch_test],
    'kbet_train': [kbet_train],
    'kbet_val': [kbet_val],
    'kbet_test': [kbet_test]
})

#%%

# Append the dataframe to a text file
results_file = f'{outdir}/benchmark_real_data_test.txt'
header = not os.path.exists(results_file)
results_df.to_csv(results_file, mode='a', header=header, index=False, sep='\t')

# %%
