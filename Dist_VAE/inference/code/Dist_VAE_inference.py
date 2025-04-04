
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse
import sys
import matplotlib.pyplot as plt
import json
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, accuracy_score, precision_score, roc_curve, roc_auc_score,auc,precision_recall_curve, average_precision_score
from itertools import cycle
import anndata as ad
import random

def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')


parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")

# Add arguments
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--Dist_VAE_DAL_model_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data', help="Path to open conference data.")
parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")
parser.add_argument("--model_inference_parameters_file", type=str, default='model_parameters.pth', help="File name for storing model parameters.")
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--model_configuration_path', type=str, default='/home/luozeyu/desktop/big_model/',
                    help=' system path add for import model')
parser.add_argument('--epoch_start_for_loss_plot_only', type=int, default=1, help='Epoch start for loss plot only')
parser.add_argument('--device_choose', type=str, default='cpu',
                    help="Select the device to run the model on (either 'cpu' or 'cuda:<device_number>')")

parser.add_argument('--label_mapping', type=str, default='{"tos": 0, "toR": 1}',
                    help='JSON string for label mapping e.g. \'{"tos": 0, "toR": 1}\'')
parser.add_argument('--class_num', type=int, default=2, help='Number of classes.')
parser.add_argument('--drug_label_choose', type=str, default='label', help='label of classes.')
parser.add_argument('--seed_set', type=int, default=42, help='Number of random seed.')
parser.add_argument('--AUC_threhold', type=float, default=0.05 )

parser.add_argument("--inference_only", type=strict_str2bool, default=False, help="if only inference and not conduct evalution.")



# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
open_path = args.open_path
save_path = args.save_path
file_prefix = args.file_prefix
model_inference_parameters_file = args.model_inference_parameters_file
batch_size = args.batch_size
path_to_add = args.model_configuration_path
epoch_start_for_loss_plot_only = args.epoch_start_for_loss_plot_only
device_choose = args.device_choose


label_mapping = json.loads(args.label_mapping)
class_num = args.class_num
drug_label_choose = args.drug_label_choose

if args.inference_only:
    drug_label_choose = None




seed_set = args.seed_set

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(seed_set)


print(f"Open Path: {open_path}")
print(f"Save Path: {save_path}")
print(f"File Prefix: {file_prefix}")
print(f"Model Parameters File: {model_inference_parameters_file}")
print(f"Epoch Start for Loss Plot Only: {epoch_start_for_loss_plot_only}")
print(f'path_to_add:{path_to_add}')
print("Label Mapping:", label_mapping)
print("Number of Classes (Expected):", class_num)

## create save path
if not os.path.exists(save_path):
    os.makedirs(save_path)

#########Add the directory containing the model definition to the system path to enable the VAE pretraining encoder call
if path_to_add not in sys.path:
    sys.path.append(path_to_add)

from distillation_VAE_pretraining_model.config import config

input_dim = config['input_dim']
hidden_dim_layer0 = config['hidden_dim_layer0']
Encoder_layer_dims = config['Encoder_layer_dims']
Decoder_layer_dims = config['Decoder_layer_dims']
hidden_dim_layer_out_Z = config['hidden_dim_layer_out_Z']
z_dim = config['z_dim']

# Choose device based on the argument
device = torch.device(args.device_choose if (torch.cuda.is_available() and "cuda" in args.device_choose) else "cpu")
print(f"Using device: {device}")


############################################
from distillation_VAE_pretraining_model.Dist_VAE_model_inference import Swish, VAEclassification, modelinference, modelevalution



## define label mapping function
def parse_label_mapping(label_mapping):

    sorted_labels = sorted(label_mapping.items(), key=lambda item: item[1])
    class_labels = [item[0] for item in sorted_labels]

    return class_labels



##input_data
filenames = os.listdir(open_path)
for filename in filenames:

    full_path = os.path.join(open_path, filename)

    # Get the file extension
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
        scFoundation_Embedding = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X  # 确保是NumPy数组
        print("Loaded .h5ad file:", scFoundation_Embedding)
    else:
        print("Unsupported file format")

# Check if the data contains NaN values
print("DATA is containing NA?: ", np.isnan(scFoundation_Embedding).any())
print('Embedding data shape: ',scFoundation_Embedding.shape)
print('info data shape: ',scFoundation_Embedding_info.shape)

torch.autograd.set_detect_anomaly(True)




scores = []
if not args.inference_only:

    scFoundation_Embedding_info[drug_label_choose] = scFoundation_Embedding_info[drug_label_choose].map(label_mapping)

    # check data and label if correctly
    actual_class_num = scFoundation_Embedding_info[drug_label_choose].nunique()
    print("Number of classes (Actual):", actual_class_num)


    if actual_class_num != class_num:
        raise ValueError(f"Error: The actual number of classes {actual_class_num} does not match the expected {class_num}.")


    ## prepare data for inference
    X_inference = scFoundation_Embedding
    y_inference = scFoundation_Embedding_info[drug_label_choose]

    X_inference_tensor = torch.Tensor(X_inference).to(device)
    y_inference_tensor = torch.LongTensor(y_inference.values).to(device)
    inference_dataset = TensorDataset(X_inference_tensor, y_inference_tensor)
    inference_loader = DataLoader(inference_dataset,batch_size=batch_size, shuffle=False,
                            drop_last=False)
else:
    # Inference-only mode (no labels)
    print("Inference-only mode: Labels will not be included.")

    # Prepare data for inference (without labels)
    X_inference = scFoundation_Embedding
    X_inference_tensor = torch.Tensor(X_inference).to(device)

    inference_dataset = TensorDataset(X_inference_tensor)  # No labels for inference
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

##define model

load_model = VAEclassification(Encoder_layer_dims,hidden_dim_layer_out_Z, z_dim,class_num).to(device)
load_model.load_state_dict(torch.load(os.path.join(args.Dist_VAE_DAL_model_path, model_inference_parameters_file), map_location=device))

#fix Encoder paramter
for param in load_model.Encoder_resblocks.parameters():
    param.requires_grad = False
for param in load_model.fc21.parameters():
    param.requires_grad = False
for param in load_model.fc22.parameters():
    param.requires_grad = False


# calculate matrixs

auc_scores = []
mcc_scores = []
f1_scores = []
recall_scores = []
accuracy_scores = []
precision_scores = []



if not args.inference_only:
    predictions, targets = modelevalution(load_model, inference_loader)

    targets = [int(x) for x in targets]

    predictions_prob = [float(x) for x in predictions]
    predictions_label = [1 if x > args.AUC_threhold else 0 for x in predictions_prob]


    current_auc = roc_auc_score(targets, predictions_prob)
    current_mcc = matthews_corrcoef(targets, predictions_label)
    current_f1 = f1_score(targets, predictions_label)
    current_recall = recall_score(targets, predictions_label)
    current_accuracy = accuracy_score(targets, predictions_label)
    current_precision = precision_score(targets, predictions_label)
    auc_scores.append(current_auc)
    mcc_scores.append(current_mcc)
    f1_scores.append(current_f1)
    recall_scores.append(current_recall)
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


    ##  evalution matrixs save
    results_df = pd.DataFrame({
        "AUC": auc_scores,
        "MCC": mcc_scores,
        "F1 Score": f1_scores,
        "Recall": recall_scores,
        "Accuracy": accuracy_scores,
        "Precision": precision_scores
    })

    results_df.to_excel(os.path.join(save_path, f"{file_prefix}evaluation_metrics.xlsx"), index=False)


    ## calculate matrixs for auc and PR curve plot
    fpr, tpr, _ = roc_curve(targets, predictions_prob)

    precision, recall, _ = precision_recall_curve(targets, predictions_prob)
    ap_score = average_precision_score(targets, predictions_prob)



    # Initialize the plot
    plt.figure(figsize=(8, 6))

    # Plot the ROC curve for each fold

    plt.plot(fpr, tpr, color='#1f77b4', lw=1.5, label=f'ROC (AUC = {auc_scores[0]:.4f})')

    # # Compute the mean ROC curve (averaging the folds)

    # Plot the diagonal line for random predictions
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)

    # auc figure save
    plt.savefig(os.path.join(save_path,f"{file_prefix}inference_auc.pdf"),dpi = 1000, format='pdf')



    ##PR plot

    # Set custom colors for plotting (using a clean scientific color palette)
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])  # Blue, Orange, Green, common in SCI-style plots

    # Initialize the plot
    plt.figure(figsize=(8, 6))

    # Plot the ROC curve for each fold

    plt.plot(recall, precision,ap_score, color='#1f77b4', lw=1.5,label='PR Curve (AP={:0.4f})'.format(ap_score))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="upper right", fontsize=10)
    # save PR figure
    plt.savefig(os.path.join(save_path,f"{file_prefix}inference_PR.pdf"),dpi = 1000, format='pdf')



    class_labels = parse_label_mapping(label_mapping)
    print("Class Labels:", class_labels)

    value_to_label = {v: k for k, v in label_mapping.items()}

    mapped_targets = [value_to_label[x] for x in targets]
    mapped_predictions = [value_to_label[x] for x in predictions_label]



    prediction_df = pd.DataFrame({
        'True Label': targets,
        'Predicted Label': mapped_predictions,
        'Predicted Probability': predictions_prob
    })


    prediction_df.to_excel(os.path.join(save_path, f'{file_prefix}_True_and_inference_label_prob_results.xlsx'), index=False)

else:
    predictions = modelinference(load_model, inference_loader)
    predictions_prob = [float(x) for x in predictions]
    predictions_label = [1 if x > args.AUC_threhold else 0 for x in predictions_prob]

    value_to_label = {v: k for k, v in label_mapping.items()}
    mapped_predictions = [value_to_label[x] for x in predictions_label]
    prediction_df = pd.DataFrame({
        'Predicted Label': mapped_predictions,
        'Predicted Probability': predictions_prob
    })

    prediction_df.to_excel(os.path.join(save_path, f'{file_prefix}_inference_label_prob_results.xlsx'), index=False)
