# -*- coding: utf-8 -*-
"""
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
import anndata as ad

"""
Revised IntegratedGradientsCalculator class
"""

from captum.attr import GradientShap
from protloc_mex1.SHAP_plus import SHAP_importance_sum

class CaptumShapCalculatorCPU:

    def __init__(self, model, X_input, X_input_tensor, batch_size=100, n_samples=100):
        self.model = model
        self.X_input = X_input
        self.X_input_tensor = X_input_tensor
        self.batch_size = batch_size
        self.n_samples = n_samples

    def _compute_shap_values_define(
            self,
            target_type,
            baseline_type,
            random_baseline_size,
            alpha,
            stdevs
    ):
        """
        Compute SHAP values with support for multiple baseline strategies.

        Parameters:
        -------
        target_type: int
            Target index for model output or other available target;
            For multi-output models, specify the output channel/dimension of interest.
        baseline_type: str, options ['zero', 'mean', 'random', 'mix']
            - 'zero':   Use zero vector as baseline
            - 'mean':   Use mean of X_input_tensor as baseline
            - 'random': Randomly sample from X_input_tensor and take mean as baseline
            - 'mix':    Weighted combination of mean and random baselines using alpha
        random_baseline_size: int
            Number of samples for random sampling in 'random' or 'mix' baseline types
        alpha: float
            Weighting coefficient for mean_baseline and random_baseline when using 'mix'
        stdevs: float
            Noise scale parameter for GradientShap in captum

        Returns:
        -------
        all_shap_values: np.ndarray
            SHAP values with same shape as X_input_tensor
        all_deltas: np.ndarray
            Convergence indicator array output by captum, same number as samples
        """

        device = self.X_input_tensor.device

        # ============ 1) Baseline generation logic ============
        # Note: captum's baselines can have the same shape as X_batch, or be single/multiple samples.
        # Specific usage can be adjusted based on requirements.
        if baseline_type == 'zero':
            # Zero baseline with same size as X_full
            baseline_full = torch.zeros_like(self.X_input_tensor).to(device)

        elif baseline_type == 'mean':
            # Average over dim=0, get (1, C, H, W) or (1, D)
            # Then expand to same length as X_full (N, ...) for subsequent batch indexing
            mean_vec = self.X_input_tensor.mean(dim=0, keepdim=True).to(device)
            # Broadcast (1, C, H, W) to (N, C, H, W)
            baseline_full = mean_vec.expand_as(self.X_input_tensor)

        elif baseline_type == 'random_mean':
            # Randomly select random_baseline_size samples within [0, N) range
            N = self.X_input_tensor.shape[0]
            idx = torch.randint(low=0, high=N, size=(random_baseline_size,))
            random_samples = self.X_input_tensor[idx].to(device)
            # Take mean of these samples
            random_baseline = random_samples.mean(dim=0, keepdim=True)
            # Expand to (N, C, H, W)
            baseline_full = random_baseline.expand_as(self.X_input_tensor)

        elif baseline_type == 'random':
            # 1) Randomly select random_baseline_size samples within [0, N) range
            N = self.X_input_tensor.shape[0]
            idx = torch.randint(low=0, high=N, size=(random_baseline_size,))
            random_samples = self.X_input_tensor[idx].to(device)
            # 2) Use random_samples directly as multiple baselines instead of taking mean
            baseline_full = random_samples

        elif baseline_type == 'mix':
            # mean baseline
            mean_vec = self.X_input_tensor.mean(dim=0, keepdim=True).to(device)
            # random baseline
            N = self.X_input_tensor.shape[0]
            idx = torch.randint(low=0, high=N, size=(random_baseline_size,))
            random_samples = self.X_input_tensor[idx].to(device)
            random_baseline = random_samples.mean(dim=0, keepdim=True)

            # Alpha weighting
            mix_baseline = alpha * mean_vec + (1.0 - alpha) * random_baseline
            # Expand to (N, C, H, W)
            baseline_full = mix_baseline.expand_as(self.X_input_tensor)

        else:
            raise ValueError(f"Unknown baseline_type: {baseline_type}")

        # ============ 2) Define wrapper forward function, only return main model output ============
        # If model's forward method returns multiple values, define a wrapper function to return only the output
        def custom_forward(x):
            output = self.model(x)
            if isinstance(output, tuple):
                return output[0]  # Return first output
            else:
                return output

        # ============ 3) Use captum's GradientShap ============
        shap_calculator = GradientShap(custom_forward)

        all_shap_values = []
        all_deltas = []

        # Calculate required number of batches
        num_batches = (self.X_input_tensor.shape[0] + self.batch_size - 1) // self.batch_size

        # ============ 4) Batch computation of SHAP ============
        for batch_idx in tqdm(range(num_batches), desc="Computing SHAP Values"):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, self.X_input_tensor.shape[0])
            X_batch = self.X_input_tensor[start_idx:end_idx]
            if baseline_type == 'random':
                # In multi-baseline case, pass entire baseline_dist directly
                baseline_batch = baseline_full
            else:
                # In single baseline case, pass baseline corresponding to current batch
                baseline_batch = baseline_full[start_idx:end_idx]

            # Compute SHAP values using captum's baseline and n_samples parameters
            attributions, delta = shap_calculator.attribute(
                X_batch,
                baselines=baseline_batch,
                target=target_type,
                n_samples=self.n_samples,
                stdevs=stdevs,  # Adjustable standard deviation for baseline perturbation
                return_convergence_delta=True
            )

            # Convert to numpy arrays for saving
            attributions_np = attributions.detach().numpy()
            all_shap_values.append(attributions_np)

            # Assume delta shape is (n_samples * batch_size, )
            delta_np = delta.detach().numpy()
            # Reshape delta to (batch_size, n_samples)
            delta_reshaped = delta_np.reshape(-1, self.n_samples)
            # Average delta values for each sample, get shape (batch_size, )
            delta_per_sample = delta_reshaped.mean(axis=1, keepdims=True)
            all_deltas.append(delta_per_sample)

        all_shap_values = np.concatenate(all_shap_values, axis=0)
        all_deltas = np.concatenate(all_deltas, axis=0)

        return all_shap_values, all_deltas

    def compute_shap(self, type_class,baseline_type='zero',
            random_baseline_size=10,
            alpha=0.5,
            stdevs=0.09):

        self.all_shap_values = []
        self.all_deltas = []
        for i in range(len(type_class)):
            all_shap_values_inn, all_deltas_inn = self._compute_shap_values_define(i,baseline_type,
            random_baseline_size,
            alpha,
            stdevs)
            self.all_shap_values.append(all_shap_values_inn)
            self.all_deltas.append(all_deltas_inn)

        # Convert to DataFrame format
        self.all_shap_values = list(
            map(lambda x: pd.DataFrame(x, index=self.X_input.index, columns=self.X_input.columns),
                self.all_shap_values))
        self.all_deltas = list(
            map(lambda x: pd.DataFrame(x, index=self.X_input.index, columns=['approximation error']),
                self.all_deltas))

        self.all_shap_values = dict(zip(type_class, self.all_shap_values))
        self.all_deltas = dict(zip(type_class, self.all_deltas))

        return self.all_shap_values, self.all_deltas

    def shap_save(self, save_path):

        all_shap_value_save = {key: value for key, value in
                               self.all_shap_values.items()}
        for key, value in self.all_shap_values.items():
            file_path = f"{save_path}{key}_shap_values.csv"
            value.to_csv(file_path, index=True)

        for key, value in self.all_deltas.items():
            file_path = f"{save_path}{key}_approximation_error_value.csv"
            value.to_csv(file_path, index=True)

        return all_shap_value_save

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

# Add arguments
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--open_path_conference_data", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data', help="Path to open conference data.")
parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")
parser.add_argument('--path_to_add', type=str, default='/home/luozeyu/desktop/big_model/',
                    help='System path to add for importing model')

parser.add_argument("--model_parameters_file", type=str, default='model_parameters.pth', help="File name for storing model parameters.")
parser.add_argument('--device_choose', type=str, default='cuda:0', help='Device choice')
parser.add_argument('--label_mapping', type=str, default='{"tos": 0, "toR": 1}',
                    help='JSON string for label mapping e.g. \'{"tos": 0, "toR": 1}\'')
parser.add_argument('--class_num', type=int, default=2, help='Number of classes.')
parser.add_argument('--ID', type=str, default='sample_id', help='ID label.')
parser.add_argument('--seed_set', type=int, default=42, help='Random seed number.')

parser.add_argument(
        "--baseline_type",
        type=str,
        choices=["zero", "mean", "random", "mix"],
        default="random",
        help="Baseline type: 'zero', 'mean', 'random', or 'mix'. Default is 'random'."
    )

parser.add_argument('--random_baseline_sample', type=int, default=10, help='Number of random baseline samples to choose.')

parser.add_argument("--shap_plot_all", type=strict_str2bool, default=False, help="Enable SHAP plotting for all classes.")

parser.add_argument(
    "--shap_calculate_type",
    type=str,
    nargs='+',  # Allows multiple values
    default=["fa34os", "fa34oc"],  # Default includes both values
    help="Select calculation types; multiple values can be included, e.g., fa34os fa34oc"
)

parser.add_argument(
    "--shap_importance_plot_figure_size",
    type=int,
    nargs=2,
    default=[15, 10],
    help="Set the figure size for the SHAP importance plot as two integers: width height (e.g., 15 10)"
)

parser.add_argument(
    "--shap_summary_plot_figure_size",
    type=int,
    nargs=2,
    default=[10, 15],
    help="Set the figure size for the SHAP summary plot as two integers: width height (e.g., 10 15)"
)

parser.add_argument(
    "--feature_attribution_display_num",
    type=int,
    default=30
)

# Add cmap_color parameter
parser.add_argument('--cmap_color', type=str, default='Spectral',
                        help='Color map name for visualization, e.g.: YlGnBu, viridis, plasma, Spectral, etc. Default is Spectral')

parser.add_argument('--feature_name_to_gene', type=str)

# Parse the arguments
args = parser.parse_args()

# Set PDF font parameters
plt.rcParams['pdf.fonttype'] = 42  # Ensure fonts are embedded rather than converted to paths

def set_seed(seed):
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
    sys.path.append(args.path_to_add)


from Dist_VAE.distillation_VAE_pretraining_model.config import config


# Working area
# Parse label mapping string to dictionary
label_mapping = json.loads(args.label_mapping)

print(f"Gene ID: {args.ID}")
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

device = torch.device(args.device_choose if torch.cuda.is_available() else "cpu")
print("Using device:", device)

############################################
import torch
from torch import nn
from torch.nn import functional as F

class Swish(nn.Module):
    def __init__(self, trainable_beta=False, initial_beta=1.0):
        super(Swish, self).__init__()
        if trainable_beta:
            self.beta = nn.Parameter(torch.tensor(initial_beta))  # Trainable β
        else:
            self.beta = initial_beta  # Fixed β

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

# Trainable β
swish = Swish(trainable_beta=True)

class VAEclassification(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            # Apply spectral normalization first
            self.fc = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=5)
            self.fc = nn.Linear(in_dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)
            self.swish = swish

            init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')

            if in_dim != out_dim:
                self.downsample = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=5)
                self.bn = nn.BatchNorm1d(out_dim)
                init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            out = self.swish(self.bn(self.fc(x)))

            if self.downsample is not None:
                x = self.downsample(x)
            return out + x

    def __init__(self, Encoder_layer_dims, hidden_dim_layer_out_Z, z_dim, class_num):
        super(VAEclassification, self).__init__()
        # Encoder

        # Resblock
        self.Encoder_resblocks = nn.ModuleList()

        for i in range(len(Encoder_layer_dims) - 1):
            self.Encoder_resblocks.append(self.ResBlock(Encoder_layer_dims[i], Encoder_layer_dims[i + 1]))

        # Apply spectral normalization and weight normalization
        self.fc21 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim), n_power_iterations=5)  # mu layer
        init.xavier_normal_(self.fc21.weight)

        self.fc22 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim),
                                      n_power_iterations=5)  # logvariance layer
        init.xavier_normal_(self.fc22.weight)

        self.fc3 = spectral_norm(nn.Linear(z_dim, z_dim // 2), n_power_iterations=5)
        init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        self.bn3 = nn.BatchNorm1d(z_dim // 2)
        self.swish = swish
        self.fc4 = spectral_norm(nn.Linear(z_dim // 2, class_num), n_power_iterations=5)
        init.xavier_normal_(self.fc4.weight)

    def encode(self, x):
        h = x
        for block in self.Encoder_resblocks:
            h = block(h)
        return self.fc21(h), self.fc22(h)  # mu, logvariance

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def classifier(self, x):
        h = self.fc3(x)
        h = self.bn3(h)
        h = self.swish(h)
        h = self.fc4(h)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, x.shape[1]))
        z = self.reparameterize(mu, logvar)
        class_logits = self.classifier(z)
        return class_logits,z

# Load model

# Input data
filenames = os.listdir(args.open_path)
for filename in filenames:
    full_path = os.path.join(args.open_path, filename)

    # Get file extension
    _, file_extension = os.path.splitext(filename)

    # Load data based on different file extensions
    if file_extension == '.npy':
        scFoundation_Embedding = np.load(full_path)
        print("Loaded .npy file:", scFoundation_Embedding)
    elif file_extension == '.csv':
        scFoundation_Embedding_info = pd.read_csv(full_path)
        print("Loaded .csv file:", scFoundation_Embedding_info)
    elif file_extension == '.xlsx':
        scFoundation_Embedding_info = pd.read_excel(full_path)
        print("Loaded .xlsx file:", scFoundation_Embedding_info)
    elif file_extension == '.h5ad':
        adata = ad.read_h5ad(full_path)
        scFoundation_Embedding = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X  # Ensure NumPy array
        print("Loaded .h5ad file:", scFoundation_Embedding)
    else:
        print("Unsupported file format")

# Check if data contains NaN
print("DATA is containing NA?: ", np.isnan(scFoundation_Embedding).any())
print('Embedding data shape: ',scFoundation_Embedding.shape)

# No labels and ID, can be read after model_inference

num_features = scFoundation_Embedding.shape[1]

# Read Excel file using absolute path
file_path = args.feature_name_to_gene

data_feature_to_gene = pd.read_excel(file_path)

# Generate feature_names list from first column
feature_names = data_feature_to_gene.iloc[:, 0].tolist()

# Create the DataFrame
X_input = pd.DataFrame(scFoundation_Embedding,  columns=feature_names)

# Set the name of the index
X_input.index.name = args.ID

# Model
input_dim = scFoundation_Embedding.shape[1]

model = VAEclassification(Encoder_layer_dims,hidden_dim_layer_out_Z, z_dim, args.class_num).to(device)

model.load_state_dict(torch.load(os.path.join(args.open_path_conference_data, args.model_parameters_file), map_location=device))

model.eval()  # Set model to evaluation mode

# Convert predicted labels
inv_label_mapping = {v: k for k, v in label_mapping.items()}

# Get number of classes
num_classes = len(inv_label_mapping)

# Generate class list in class index order
type_class_list = [inv_label_mapping[i] for i in range(num_classes)]

# Print class list for confirmation
print("Class list:", type_class_list)

# Start local analysis of PyTorch-based DNN model with Deep SHAP on training set

X_input_tensor= torch.Tensor(scFoundation_Embedding).to(device)

if args.device_choose == 'cpu':
    print('using cpu')

Gradientshap =CaptumShapCalculatorCPU(model,X_input, X_input_tensor,batch_size=100, n_samples=100)

Gradientshap_test_values = Gradientshap.compute_shap(type_class_list,
                                                     baseline_type=args.baseline_type,
                                                     random_baseline_size=args.random_baseline_sample,
                                                     alpha=0.5,
                                                     stdevs=0.09)

Gradientshap_test_values_save = Gradientshap.shap_save(save_path=args.save_path+"/"+ f"{args.file_prefix}_")

shap_test_values=Gradientshap_test_values[0]

print('run feature_aggregation_value_conduct_DNN_IG.py success, feature aggregation value and prediction outcome are deploted in ./output')

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
    
# Remove prediction labels and original data labels for categories not needed for research
if shap_plot:
    
    shap_test_values_select =shap_test_values

    if args.shap_plot_all:
        for i,value in enumerate(type_class_list):
            if value not in args.shap_calculate_type:
                del shap_test_values_select[value]
    else:
        print('all classes will be used to drawing map ')

    # SHAP global importance analysis
    test_shap_plot=FeaturePlot(X_input,shap_test_values_select)
    test_shap_plot.shap_importance_plot(file_name= f'{args.file_prefix}',save_path=args.save_path+"/",plot_size=args.shap_importance_plot_figure_size)

    # The following model analysis needs to consider sample classification issues

    # SHAP global summary_plot analysis (note that classes need to be separated here)
    # Create color map with gradient from yellow to purple
    cmap = plt.cm.get_cmap(f"{args.cmap_color}")

    test_shap_plot.Shapley_summary_plot(file_name=f'_{args.file_prefix}',
                                        save_path=args.save_path+"/",
                                        plot_size=args.shap_summary_plot_figure_size,
                                        max_display=args.feature_attribution_display_num,cmap="viridis" )

    print('shap plot finished, plot outcome are deploted in ./output')

# Feature importance calculation
all_data=SHAP_importance_sum()
all_data.shapley_data_all=Gradientshap_test_values_save
   
# Calculate all_data
SHAP_importance_sum_calculate_outcome=all_data.IG_importance_sum_claulate_process(depleted_ID_len=0,file_name=args.file_prefix+'_')

SHAP_importance_sum_calculate_outcome['shap_feature_importance'].to_csv(args.save_path+"/"+args.file_prefix+'_'+"shap_feature_importance.csv",index_label='feature_name')

test_sum = SHAP_importance_sum_calculate_outcome['shap_feature_importance'].sum(axis=1)
test_sum_df = pd.DataFrame(test_sum, columns=[f"{args.file_prefix}_sum"])
num_classes = len(inv_label_mapping)

# 按照类别索引顺序生成类别列表
type_class_list = [inv_label_mapping[i] for i in range(num_classes)]


# 打印类别列表以确认
print("类别列表:", type_class_list)


##开始进行训练集Deep SHAP对基于pytorchd的DNN的模型进行局部分析


X_input_tensor= torch.Tensor(scFoundation_Embedding).to(device)

# if args.device_choose == 'cpu':
#     print('using cpu')
#     Gradientshap =CaptumShapCalculatorCPU(model,X_input, X_input_tensor,batch_size=100, n_samples=100)
# else:
#     print('using GPU')
#     Gradientshap = CaptumShapCalculatorCPU(model, X_input, X_input_tensor, batch_size=100, n_samples=100)

if args.device_choose == 'cpu':
    print('using cpu')

Gradientshap =CaptumShapCalculatorCPU(model,X_input, X_input_tensor,batch_size=100, n_samples=100)

Gradientshap_test_values = Gradientshap.compute_shap(type_class_list,
                                                     baseline_type=args.baseline_type,
                                                     random_baseline_size=args.random_baseline_sample,
                                                     alpha=0.5,
                                                     stdevs=0.09)


Gradientshap_test_values_save = Gradientshap.shap_save(save_path=args.save_path+"/"+ f"{args.file_prefix}_")


shap_test_values=Gradientshap_test_values[0]
# print(shap_test_values)

print('run feature_aggregation_value_conduct_DNN_IG.py success, feature aggregation value and prediction outcome are deploted in ./output')




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
    

##将predict标签和原始数据标签不需要研究的种类删除
if shap_plot:
    
    # shap_test_values_select =dict(zip(model.classes_,shap_test_values))
    shap_test_values_select =shap_test_values

    if args.shap_plot_all:
        for i,value in enumerate(type_class_list):
            if value not in args.shap_calculate_type:
                del shap_test_values_select[value]
    else:
        print('all classes will be used to drawing map ')


##shap全局importance分析

    test_shap_plot=FeaturePlot(X_input,shap_test_values_select)
    test_shap_plot.shap_importance_plot(file_name= f'{args.file_prefix}',save_path=args.save_path+"/",plot_size=args.shap_importance_plot_figure_size)


##以下模型分析需要考虑样本分类问题

##shap全局summary_plot分析(注意这里需要分class了)
#create color map with gradient from yellow to purple
    cmap = plt.cm.get_cmap(f"{args.cmap_color}")
# cmap = plt.cm.get_cmap("cividis")

    test_shap_plot.Shapley_summary_plot(file_name=f'_{args.file_prefix}',
                                        save_path=args.save_path+"/",
                                        plot_size=args.shap_summary_plot_figure_size,
                                        max_display=args.feature_attribution_display_num,cmap="viridis" )

    print('shap plot finished, plot outcome are deploted in ./output')





##特征重要性计算

all_data=SHAP_importance_sum()
all_data.shapley_data_all=Gradientshap_test_values_save

   
##对all_data进行计算
SHAP_importance_sum_calculate_outcome=all_data.IG_importance_sum_claulate_process(depleted_ID_len=0,file_name=args.file_prefix+'_')


SHAP_importance_sum_calculate_outcome['shap_feature_importance'].to_csv(args.save_path+"/"+args.file_prefix+'_'+"shap_feature_importance.csv",index_label='feature_name')


test_sum = SHAP_importance_sum_calculate_outcome['shap_feature_importance'].sum(axis=1)
test_sum_df = pd.DataFrame(test_sum, columns=[f"{args.file_prefix}_sum"])
test_sum_df['Rank_Dense'] = test_sum_df[f"{args.file_prefix}_sum"].rank(method='dense', ascending=False)
test_sum_df['Rank_Average'] = test_sum_df[f"{args.file_prefix}_sum"].rank(method='average', ascending=False)
test_sum_df.to_csv(f"{args.save_path}/{args.file_prefix}_importance_Rank.csv", index=True, header=True)


print('run feature_aggregation_value_conduct_Tree_SHAP.py success, feature importance value and importance rank are deploted in ./output')
