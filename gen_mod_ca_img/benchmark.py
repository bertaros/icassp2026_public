#%%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import sem, t

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

# boxplots per noise
def boxplots_per_noise(x,y):
    
    noise_levels = x['fluo_noise'].unique()
    lat_levels = x['latent_dim'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

    for i, noise in enumerate(noise_levels):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        x_noise = x[x['fluo_noise'] == noise]
        sns.boxplot(x='model', 
                    y=y, 
                    hue='latent_dim', 
                    data=x_noise, 
                    showfliers=False, 
                    palette=sns.color_palette("ch:s=.25,rot=-.25",len(lat_levels)),
                    ax=ax)
        ax.set_title(f'Noise Level: {noise}')
        ax.set_xlabel('Model')
        if col == 0:
            ax.set_ylabel(y)
        else:
            ax.set_ylabel('')
        ax.legend_.remove()  # Remove individual legends
        ax.tick_params(axis='x', rotation=90)

    # Hide any unused subplots
    for i in range(len(noise_levels), 4):
        row, col = divmod(i, 2)
        fig.delaxes(axes[row, col])

    # Add a single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Latent Dimension', loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout()
    plt.show()
    
    return fig

def scatter_with_ci(df, metric_x, metric_y, conf=0.95, noise_level=None):
    
    grouped = df.groupby(['model', 'latent_dim'], observed=True)
    df_mean = grouped.mean(numeric_only=True).reset_index()

    # Compute standard error and confidence intervals
    n = grouped.size().values
    se_x = grouped[metric_x].sem().values
    se_y = grouped[metric_y].sem().values
    h_x = se_x * t.ppf((1 + conf) / 2., n - 1)
    h_y = se_y * t.ppf((1 + conf) / 2., n - 1)

    df_mean[f'{metric_x}_ci'] = h_x
    df_mean[f'{metric_y}_ci'] = h_y

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df_mean,
        x=metric_x,
        y=metric_y,
        hue='model',
        style='latent_dim',
        palette="pastel",
        s=100,
        alpha=0.8,
        ax=ax
    )

    # Add error bars
    for _, row in df_mean.iterrows():
        ax.errorbar(
            row[metric_x], row[metric_y],
            xerr=row[f'{metric_x}_ci'], yerr=row[f'{metric_y}_ci'],
            fmt='none', ecolor='black', alpha=0.2, capsize=3, zorder=0
        )

    ax.set_title(f'Model Performance (Mean ± 95% CI): {metric_x} vs {metric_y}, Fluo Noise: {noise_level}' if noise_level else f'Model Performance (Mean ± 95% CI): {metric_x} vs {metric_y}')
    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    ax.legend(title='Model / Latent Dim', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    
    return fig

base_results_dir = '/pool01/projects/abante_lab/snDGM/mlcb_2025/results/'

## EFFA
with open(os.path.join(base_results_dir, 'round_3/results_summary_effa_time_test_group_23_no_FS.txt'), 'r') as f:
    df_effa = pd.read_csv(f, sep='\t')
    df_effa['model'] = 'BFA'

## VAE
with open(os.path.join(base_results_dir, 'round_3/results_summary_model_FixedVarMlpVAE_test_group_23_no_FS.txt'), 'r') as f:
    df_vae = pd.read_csv(f, sep='\t')
    df_vae['model'] = 'VAE'

## SVAE
with open(os.path.join(base_results_dir, 'round_3/results_summary_model_FixedVarSupMlpVAE_test_group_23_no_FS.txt'), 'r') as f:
    df_svae = pd.read_csv(f, sep='\t')
    df_svae['model'] = 'SVAE'

## GPVAE
with open(os.path.join(base_results_dir, 'round_3/results_summary_model_SupMlpGpVAE_test.txt'), 'r') as f:
    df_gpvae = pd.read_csv(f, sep='\t')
    df_gpvae['model'] = 'GPVAE'

# For each latent dim, seed and fluo_noise, keep the best value of num_ind_pts based on kbet_test
df_gpvae_filt = df_gpvae.loc[df_gpvae.groupby(['latent_dim', 'seed', 'fluo_noise'])['kbet_test'].idxmin()].reset_index(drop=True)

df_group = pd.concat([df_effa, df_vae, df_svae, df_gpvae_filt], axis=0).reset_index(drop=True)

#%%
################################################################################################################
# Plots with Groups 2,3 and no FS neurons
################################################################################################################

# order models
df_group['model'] = pd.Categorical(
    df_group['model'], 
    categories=['BFA', 'VAE', 'SVAE', 'GPVAE'], 
    ordered=True
)

# filter out noise levels 0 and 2.5
df_group = df_group[df_group['fluo_noise'].isin([0.5, 1.0, 1.5, 2.0])]

# latent dims
df_group = df_group[df_group['latent_dim'].isin([4, 8, 16, 32, 128, 256])]
lat_levels = df_group['latent_dim'].unique()

# F = boxplots_per_noise(df, 'kbet_val')

noise_level = 1.5

df_fig = df_group[df_group['fluo_noise'].isin([noise_level])] # .isin([1])]
df_fig = df_fig[df_fig['model'].isin(['BFA', 'VAE', 'SVAE', 'GPVAE'])]
df_fig['model'] = pd.Categorical(df_fig['model'], categories=['BFA', 'VAE', 'SVAE', 'GPVAE'], ordered=True)

grouped = df_group.groupby(['model', 'latent_dim'], observed=True)
df_mean = grouped.mean(numeric_only=True).reset_index()

# scatter plots
p = scatter_with_ci(df_fig, 'sil_test', 'kbet_test', noise_level=noise_level)
p = scatter_with_ci(df_fig, 'firing_acc', 'kbet_test')
p = scatter_with_ci(df_fig, 'ari_test', 'kbet_test')
p = scatter_with_ci(df_fig, 'mae_x_test', 'kbet_test')
p = scatter_with_ci(df_fig, 'med_ent_batch_test', 'med_ent_firing_test')
# # p.savefig(f'{results_dir}/scatter_silval_kbetval.pdf', bbox_inches='tight')

# mae vs latent dim
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x='model', y='mae_x_test', hue='latent_dim', data=df_fig, showfliers=False, 
            palette=sns.color_palette("ch:s=.25,rot=-.25",len(lat_levels)), ax=ax)
fig.tight_layout()
# fig.savefig(f'{results_dir_2}/boxplot_mae.pdf', bbox_inches='tight')
plt.show()

# kbet vs latent dim
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x='model', y='kbet_test', hue='latent_dim', data=df_fig, showfliers=False, 
            palette=sns.color_palette("ch:s=.25,rot=-.25",len(lat_levels)), ax=ax)
fig.tight_layout()
# fig.savefig(f'{results_dir_2}/boxplot_ari.pdf', bbox_inches='tight')
plt.show()

#%%
metric = 'kbet_test'
best_latent_dims = {}
for model in df_group['model'].unique():
    df_model = df_group[df_group['model'] == model]
    best_latent_dims[model] = {}
    for noise in df_model['fluo_noise'].unique():
        df_noise = df_model[df_model['fluo_noise'] == noise]
        mean_kbet = df_noise.groupby('latent_dim')[metric].mean()
        best_latent_dim = mean_kbet.idxmin()  # Get the latent_dim with the lowest mean kbet_val
        best_latent_dims[model][noise] = best_latent_dim

# Do a single boxplot of kbet_val for each model, using only the best latent dimension for each noise level, color by noise level
df_best = pd.DataFrame()
for model in df_group['model'].unique():
    for noise in df_group['fluo_noise'].unique():
        best_latent_dim = best_latent_dims[model][noise]
        df_subset = df_group[(df_group['model'] == model) & (df_group['fluo_noise'] == noise) & (df_group['latent_dim'] == best_latent_dim)]
        df_best = pd.concat([df_best, df_subset], axis=0)

fig, ax = plt.subplots(figsize=(8, 6))
# Use the full viridis colormap
palette = sns.color_palette("viridis", as_cmap=True)
# Map noise levels to colors using the full colormap, but invert the order
noise_levels = sorted(df_best['fluo_noise'].unique())
colors = [palette(i / (len(noise_levels) - 1)) for i in reversed(range(len(noise_levels)))]
color_dict = dict(zip(noise_levels, colors))
sns.boxplot(
    x='model',
    y=metric,
    hue='fluo_noise',
    data=df_best,
    showfliers=False,
    palette=color_dict,
    ax=ax,
    linewidth=0.7  # Make boxplot lines thinner
)
ax.set_title('KBET Value per Model (Best Latent Dim per Noise Level)')
ax.set_xlabel('Model')
ax.set_ylabel('KBET Value')
ax.legend(title='Noise Level', bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()
plt.savefig(f'/pool01/projects/abante_lab/snDGM/icassp2026/boxplot_kbet_test_best_latent_dim_group_23_no_FS.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
