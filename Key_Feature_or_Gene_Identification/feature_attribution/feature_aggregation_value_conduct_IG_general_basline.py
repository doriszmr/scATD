# -*- coding: utf-8 -*-
"""
Feature Attribution Analysis using Integrated Gradients for Deep Neural Networks
This script performs feature importance analysis using Integrated Gradients method
on VAE-based classification models for single-cell RNA-seq data analysis.

Created on Mon Jun 17 15:44:41 2024
@author: qq102
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
from protloc_mex1.SHAP_plus import FeaturePlot
from tqdm import tqdm
from captum.attr import IntegratedGradients

from protloc_mex1.SHAP_plus import SHAP_importance_sum



# CPU version of Integrated Gradients Calculator
class IntegratedGradientsCalculatorCPU:
    """
    Integrated Gradients Calculator for CPU computation with multiple baseline options
    """
    def __init__(self, model, X_input, X_input_tensor, batch_size=100, n_steps=50):
        self.model = model
        self.X_input = X_input
        self.X_input_tensor = X_input_tensor
        self.batch_size = batch_size
        self.n_steps = n_steps

    def _compute_integrated_gradients_define(
            self,
            target_type,
            baseline_type,
            random_baseline_size,
            alpha):

        device = self.X_input_tensor.device

        # Baseline generation logic
        # Note: captum baselines can have the same shape as X_batch or be single/multiple samples
        if baseline_type == 'zero':
            # Zero baseline with same size as X_full
            baseline_full = torch.zeros_like(self.X_input_tensor).to(device)

        elif baseline_type == 'mean':
            # Mean baseline across dim=0, expand to match X_full size
            mean_vec = self.X_input_tensor.mean(dim=0, keepdim=True).to(device)
            baseline_full = mean_vec.expand_as(self.X_input_tensor)

        elif baseline_type == 'random':
            # Random baseline from randomly selected samples
            N = self.X_input_tensor.shape[0]
            idx = torch.randint(low=0, high=N, size=(random_baseline_size,))
            random_samples = self.X_input_tensor[idx].to(device)
            random_baseline = random_samples.mean(dim=0, keepdim=True)
            baseline_full = random_baseline.expand_as(self.X_input_tensor)

        elif baseline_type == 'mix':
            # Mixed baseline: weighted combination of mean and random
            mean_vec = self.X_input_tensor.mean(dim=0, keepdim=True).to(device)
            N = self.X_input_tensor.shape[0]
            idx = torch.randint(low=0, high=N, size=(random_baseline_size,))
            random_samples = self.X_input_tensor[idx].to(device)
            random_baseline = random_samples.mean(dim=0, keepdim=True)

            # Alpha weighting
            mix_baseline = alpha * mean_vec + (1.0 - alpha) * random_baseline
            baseline_full = mix_baseline.expand_as(self.X_input_tensor)

        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")

        # Define wrapper forward function that returns only main model output
        def custom_forward(x):
            output = self.model(x)
            if isinstance(output, tuple):
                return output[0]  # Return first output
            else:
                return output

        ig = IntegratedGradients(custom_forward)

        all_attributions = []
        all_deltas = []

        # Calculate number of batches needed
        num_batches = (self.X_input_tensor.shape[0] + self.batch_size - 1) // self.batch_size

        # Batch computation of Integrated Gradients
        for batch_idx in tqdm(range(num_batches), desc="Computing Integrated Gradients"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.X_input_tensor.shape[0])
            X_batch = self.X_input_tensor[start_idx:end_idx]
            baseline_batch = baseline_full[start_idx:end_idx]  # Corresponding baseline for current batch

            attributions, delta = ig.attribute(
                X_batch,
                baseline_batch,
                target=target_type,
                return_convergence_delta=True,
                n_steps=self.n_steps
            )

            attributions_np = attributions.detach().numpy()
            delta_np = delta.detach().numpy()

            all_attributions.append(attributions_np)
            all_deltas.append(delta_np)

        all_attributions = np.concatenate(all_attributions, axis=0)
        all_deltas = np.concatenate(all_deltas, axis=0)

        return all_attributions, all_deltas

    def compute_integrated_gradients(self, type_class, baseline_type='zero',
            random_baseline_size=10,
            alpha=0.5):

        self.all_attributions_values = []
        self.all_deltas = []
        for i in range(len(type_class)):
            all_attributions_values_inn, all_deltas_inn = self._compute_integrated_gradients_define(i, baseline_type,
            random_baseline_size,
            alpha)
            self.all_attributions_values.append(all_attributions_values_inn)
            self.all_deltas.append(all_deltas_inn)

        self.all_attributions_values = list(
            map(lambda x: pd.DataFrame(x, index=self.X_input.index, columns=self.X_input.columns),
                self.all_attributions_values))
        self.all_deltas = list(
            map(lambda x: pd.DataFrame(x, index=self.X_input.index, columns=['approximation error']),
                self.all_deltas))

        self.all_attributions_values = dict(zip(type_class, self.all_attributions_values))
        self.all_deltas = dict(zip(type_class, self.all_deltas))

        return self.all_attributions_values, self.all_deltas

    def integrated_gradients_save(self, save_path):
        all_attributions_value_save = {key: value for key, value in
                                       self.all_attributions_values.items()}
        for key, value in self.all_attributions_values.items():
            file_path = f"{save_path}{key}_integrated_gradients_value.csv"
            value.to_csv(file_path, index=True)

        for key, value in self.all_deltas.items():
            file_path = f"{save_path}{key}_approximation_error_value.csv"
            value.to_csv(file_path, index=True)

        return all_attributions_value_save


import argparse
import torch.nn.init as init
from torch.nn.utils import spectral_norm
import json
import numpy as np
import random

parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")

def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')

# Add command line arguments
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--open_path_conference_data", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data', help="Path to open conference data.")
parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")
parser.add_argument('--path_to_add', type=str, default='/home/luozeyu/desktop/big_model/',
                    help='System path add for import model')
parser.add_argument("--model_parameters_file", type=str, default='model_parameters.pth', help="File name for storing model parameters.")

parser.add_argument('--label_mapping', type=str, default='{"tos": 0, "toR": 1}',
                    help='JSON string for label mapping e.g. \'{"tos": 0, "toR": 1}\'')
parser.add_argument('--class_num', type=int, default=2, help='Number of classes.')
parser.add_argument('--ID', type=str, default='sample_id', help='ID label for DataFrame index.')

parser.add_argument('--seed_set', type=int, default=42, help='Random seed number.')
parser.add_argument("--shap_plot_all", type=strict_str2bool, default=False, help="Enable all SHAP plots")
parser.add_argument("--baseline_type", type=str, choices=["zero", "mean", "random", "mix"], default="random",
        help="Baseline type: 'zero', 'mean', 'random', or 'mix'. Default is 'random'.")
parser.add_argument("--shap_calculate_type", type=str, nargs='+', default=["fa34os", "fa34oc"],
    help="Select calculation types; multiple values can be included, e.g., fa34os fa34oc")
parser.add_argument("--shap_importance_plot_figure_size", type=int, nargs=2, default=[15, 10],
    help="Set the figure size for the SHAP importance plot as two integers: width height (e.g., 15 10)")
parser.add_argument("--shap_summary_plot_figure_size", type=int, nargs=2, default=[10, 15],
    help="Set the figure size for the SHAP summary plot as two integers: width height (e.g., 10 15)")
parser.add_argument("--feature_attribution_display_num", type=int, default=30)
parser.add_argument('--cmap_color', type=str, default='Spectral',
                        help='Color map name for visualization, e.g.: YlGnBu, viridis, plasma, Spectral. Default is YlGnBu')

# Parse the arguments
args = parser.parse_args()

# Set PDF font parameters
plt.rcParams['pdf.fonttype'] = 42  # Ensure embedded fonts instead of converting to paths

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(args.seed_set)

# Check if directory exists, create if not
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

# Add model definition directory to system path for VAE pretraining encoder import
if args.path_to_add not in sys.path:
    sys.path.append(os.path.abspath(args.path_to_add))

from VAE_sf.VAE_sf_pretraining_model.config import config


# Working area
# Parse label mapping string to dictionary
label_mapping = json.loads(args.label_mapping)

print(f"ID: {args.ID}")
print(f"SHAP plots enabled: {args.shap_plot_all}")
print(f"SHAP calculate types: {args.shap_calculate_type}")
print(f"Importance plot size: {args.shap_importance_plot_figure_size}")

# Get parameters from config
input_dim = config['input_dim']
hidden_dim_layer0 = config['hidden_dim_layer0']
Encoder_layer_dims = config['Encoder_layer_dims']
Decoder_layer_dims = config['Decoder_layer_dims']
hidden_dim_layer_out_Z = config['hidden_dim_layer_out_Z']
z_dim = config['z_dim']


# Model definitions
import torch
from torch import nn
from torch.nn import functional as F

class Swish(nn.Module):
    """Swish activation function with optional trainable beta parameter"""
    def __init__(self, trainable_beta=False, initial_beta=1.0):
        super(Swish, self).__init__()
        if trainable_beta:
            self.beta = nn.Parameter(torch.tensor(initial_beta))  # Trainable beta
        else:
            self.beta = initial_beta  # Fixed beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# Trainable beta
swish = Swish(trainable_beta=True)

class VAEclassification(nn.Module):
    """VAE-based classification model with ResBlock encoder and classifier
    Note！！！ model is the combination of VAE-sf(pretraining, ContinuousResidualVAE) and (downstream training, VAEclassification)，
    the below module can running right，and equal to the VAE-sf inference model, you also can
    change this to your define model.
    """
    
    class ResBlock(nn.Module):
        """Residual block with batch normalization"""
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim)
            init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
            self.bn = nn.BatchNorm1d(out_dim)

            if in_dim != out_dim:
                self.downsample = nn.Linear(in_dim, out_dim)
                init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            out = F.leaky_relu(self.bn(self.fc(x)))

            if self.downsample is not None:
                x = self.downsample(x)
            return out + x

    def __init__(self, Encoder_layer_dims, hidden_dim_layer_out_Z, z_dim, class_num):
        super(VAEclassification, self).__init__()
        
        # Encoder ResBlocks
        self.Encoder_resblocks = nn.ModuleList()
        for i in range(len(Encoder_layer_dims) - 1):
            self.Encoder_resblocks.append(self.ResBlock(Encoder_layer_dims[i], Encoder_layer_dims[i + 1]))

        # Latent space layers
        self.fc21 = nn.Linear(hidden_dim_layer_out_Z, z_dim)  # mu layer
        init.xavier_normal_(self.fc21.weight)  # Xavier Initialization for mu layer
        self.fc22 = nn.Linear(hidden_dim_layer_out_Z, z_dim)  # logvariance layer
        init.xavier_normal_(self.fc22.weight)  # Xavier Initialization for logvar layer

        # Classifier layers
        self.fc3 = spectral_norm(nn.Linear(z_dim, z_dim // 2), n_power_iterations=5)
        init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        self.bn3 = nn.BatchNorm1d(z_dim // 2)
        self.swish = swish
        self.fc4 = spectral_norm(nn.Linear(z_dim // 2, class_num), n_power_iterations=5)
        init.xavier_normal_(self.fc4.weight)

    def encode(self, x):
        """Encode input through ResBlocks"""
        h = x
        for block in self.Encoder_resblocks:
            h = block(h)
        return self.fc21(h), self.fc22(h)  # mu, logvariance

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def classifier(self, x):
        """Classification head"""
        h = self.fc3(x)
        h = self.bn3(h)
        h = self.swish(h)
        h = self.fc4(h)
        return h

    def forward(self, x):
        """Forward pass through VAE classifier"""
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        class_logits = self.classifier(z)
        return class_logits






# Load input data
filenames = os.listdir(args.open_path)
for filename in filenames:
    full_path = os.path.join(args.open_path, filename)
    
    # Get file extension
    _, file_extension = os.path.splitext(filename)
    
    # Load data based on file extension
    if file_extension == '.npy':
        scFoundation_Embedding = np.load(full_path)
        print("Loaded .npy file:", scFoundation_Embedding)
    elif file_extension == '.csv':
        scFoundation_Embedding_info = pd.read_csv(full_path)
        print("Loaded .csv file:", scFoundation_Embedding_info)
    elif file_extension == '.xlsx':
        scFoundation_Embedding_info = pd.read_excel(full_path)
        print("Loaded .xlsx file:", scFoundation_Embedding_info)
    else:
        print("Unsupported file format")

# Check for NaN values in data
print("DATA is containing NA?: ", np.isnan(scFoundation_Embedding).any())
print('Embedding data shape: ', scFoundation_Embedding.shape)

# Create feature names and DataFrame
num_features = scFoundation_Embedding.shape[1]
feature_names = [f'scFoundation_{i}' for i in range(num_features)]
X_input = pd.DataFrame(scFoundation_Embedding, columns=feature_names)

# Set the name of the index
X_input.index.name = args.ID

device ='cpu'

# Initialize and load model
input_dim = scFoundation_Embedding.shape[1]
model = VAEclassification(Encoder_layer_dims, hidden_dim_layer_out_Z, z_dim, args.class_num).to(device)

model.load_state_dict(torch.load(os.path.join(args.open_path_conference_data, args.model_parameters_file), map_location=device))
model.eval()  # Set model to evaluation mode

# Create inverse label mapping
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# Get number of classes
num_classes = len(inv_label_mapping)

# Generate class list in order of class indices
type_class_list = [inv_label_mapping[i] for i in range(num_classes)]

# Print class list for confirmation
print("Class list:", type_class_list)

# Start Integrated Gradients analysis for DNN model
X_input_tensor = torch.Tensor(scFoundation_Embedding).to(device)



IG = IntegratedGradientsCalculatorCPU(model, X_input, X_input_tensor, batch_size=100, n_steps=100)

IG_test_values = IG.compute_integrated_gradients(type_class_list,
                                                  baseline_type=args.baseline_type,
                                                  random_baseline_size=10,
                                                  alpha=0.5)

IG_test_values_save = IG.integrated_gradients_save(save_path=args.save_path+"/"+ f"{args.file_prefix}_")

shap_test_values = IG_test_values[0]

print('run feature_aggregation_value_conduct_DNN_IG.py success, feature aggregation value and prediction outcome are deployed in ./output')

def get_user_preference():
    """Prompt user for generating SHAP plots."""
    while True:
        choice = input("Would you like to generate an importance and summary plot? Please type 'True' for yes or 'False' for no: ").strip().lower()
        if choice in ['true', 'false']:
            return choice == 'true'
        else:
            print("Invalid choice. Please type 'True' or 'False'.")

# Obtain user preference
shap_plot = get_user_preference()    

# Remove unwanted prediction labels and original data labels from analysis
if shap_plot:
    shap_test_values_select = shap_test_values

    if args.shap_plot_all:
        for i, value in enumerate(type_class_list):
            if value not in args.shap_calculate_type:
                del shap_test_values_select[value]
    else:
        print('all classes will be used to drawing map ')

    # Global SHAP importance analysis
    test_shap_plot = FeaturePlot(X_input, shap_test_values_select)
    test_shap_plot.shap_importance_plot(file_name=f'{args.file_prefix}', save_path=args.save_path+"/", plot_size=args.shap_importance_plot_figure_size)

    # Global SHAP summary plot analysis (class-specific analysis required)
    # Create color map with gradient
    cmap = plt.cm.get_cmap(f"{args.cmap_color}")

    test_shap_plot.Shapley_summary_plot(file_name=f'_{args.file_prefix}',
                                        save_path=args.save_path+"/",
                                        plot_size=args.shap_summary_plot_figure_size,
                                        max_display=args.feature_attribution_display_num, cmap="viridis")

    print('shap plot finished, plot outcome are deployed in ./output')

# Feature importance calculation
all_data = SHAP_importance_sum()
all_data.shapley_data_all = IG_test_values_save
   
# Calculate all_data
SHAP_importance_sum_calculate_outcome = all_data.IG_importance_sum_claulate_process(depleted_ID_len=0, file_name=args.file_prefix+'_')

SHAP_importance_sum_calculate_outcome['shap_feature_importance'].to_csv(args.save_path+"/"+args.file_prefix+'_'+"shap_feature_importance.csv", index_label='feature_name')

test_sum = SHAP_importance_sum_calculate_outcome['shap_feature_importance'].sum(axis=1)
test_sum_df = pd.DataFrame(test_sum, columns=[f"{args.file_prefix}_sum"])
test_sum_df['Rank_Dense'] = test_sum_df[f"{args.file_prefix}_sum"].rank(method='dense', ascending=False)
test_sum_df['Rank_Average'] = test_sum_df[f"{args.file_prefix}_sum"].rank(method='average', ascending=False)
test_sum_df.to_csv(f"{args.save_path}/{args.file_prefix}_importance_Rank.csv", index=True, header=True)

print('run feature_aggregation_value_conduct_Tree_SHAP.py success, feature importance value and importance rank are deployed in ./output')
