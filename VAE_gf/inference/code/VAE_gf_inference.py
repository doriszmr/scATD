
import numpy as np
import pandas as pd
import os
import torch

from torch.utils.data import DataLoader, TensorDataset

import torch.nn.init as init

import argparse

import sys

import matplotlib.pyplot as plt

import json
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, accuracy_score, precision_score, roc_curve, roc_auc_score,auc,precision_recall_curve, average_precision_score

from torch.nn.utils import spectral_norm



import random


current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, '../../../'))

sys.path.append(project_root)


def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')

# ---------------------------
# Argument parser

parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")

# Add arguments
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--open_path_conference_data", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data', help="Path to open conference data.")
parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")
parser.add_argument("--style_alignment_file", type=str)

parser.add_argument("--model_parameters_file", type=str, default='model_parameters.pth', help="File name for storing model parameters.")
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--model_configuration_path', type=str, default='/home/luozeyu/desktop/big_model/',
                    help=' system path add for import model')

parser.add_argument('--device_choose', type=str, default='cuda:0', help='cpu choose')
parser.add_argument('--label_mapping', type=str, default='{"tos": 0, "toR": 1}',
                    help='JSON string for label mapping e.g. \'{"tos": 0, "toR": 1}\'')
parser.add_argument('--class_num', type=int, default=2, help='Number of classes.')
parser.add_argument('--drug_label_choose', type=str, default='label', help='label of classes.')
parser.add_argument('--seed_set', type=int, default=42, help='Number of random seed.')
parser.add_argument("--inference_only", type=strict_str2bool, default=False, help="If True, only conduct inference (no evaluation with labels).")
parser.add_argument('--PP_threhold', type=float, default=0.5)

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
open_path = args.open_path
save_path = args.save_path
open_path_conference_data = args.open_path_conference_data
file_prefix = args.file_prefix
model_parameters_file = args.model_parameters_file
batch_size = args.batch_size
path_to_add = args.model_configuration_path

device_choose = args.device_choose
style_alignment_file = args.style_alignment_file
inference_only = args.inference_only 

# 解析标签映射字符串为字典
label_mapping = json.loads(args.label_mapping)
class_num = args.class_num
drug_label_choose = args.drug_label_choose
seed_set = args.seed_set

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(seed_set)



# 动态导入模块


# REC_beta = args.REC_beta

# Now you can use these variables in your script
print(f"Open Path: {open_path}")
print(f"Save Path: {save_path}")
print(f"Open Path Conference Data: {open_path_conference_data}")
print(f"File Prefix: {file_prefix}")
print(f"Model Parameters File: {model_parameters_file}")

print(f'path_to_add:{path_to_add}')
print("Label Mapping:", label_mapping)
print("Number of Classes (Expected):", class_num)

if not os.path.exists(save_path):
    os.makedirs(save_path)

##############################添加模型定义所在目录到系统路径, 实现vae_pretraning encoder调用
if path_to_add not in sys.path:
    sys.path.append(path_to_add)


from VAE_gf.VAE_gf_pretraining_model.config import config

# 获取配置中的参数
input_dim = config['input_dim']
hidden_dim_layer0 = config['hidden_dim_layer0']
Encoder_layer_dims = config['Encoder_layer_dims']
Decoder_layer_dims = config['Decoder_layer_dims']
hidden_dim_layer_out_Z = config['hidden_dim_layer_out_Z']
z_dim = config['z_dim']

device = torch.device(device_choose if torch.cuda.is_available() else "cpu")
print("Using device:", device)


############################################
import torch
from torch import nn
from torch.nn import functional as F

class Swish(nn.Module):
    def __init__(self, trainable_beta=False, initial_beta=1.0):
        super(Swish, self).__init__()
        if trainable_beta:
            self.beta = nn.Parameter(torch.tensor(initial_beta))  # 可训练的β
        else:
            self.beta = initial_beta  # 固定的β

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
# 可训练的β
swish = Swish(trainable_beta=True)


class VAEclassification(nn.Module):
    class ResBlock(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            # 先应用谱正则化
            self.fc = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=5)
            # self.fc = nn.Linear(in_dim, out_dim)
            self.bn = nn.BatchNorm1d(out_dim)
            self.swish = Swish(trainable_beta=True)  # 为每个 ResBlock 实例化独立的 Swish

            init.kaiming_normal_(self.fc.weight_orig, nonlinearity='leaky_relu')

            if in_dim != out_dim:

                self.downsample = spectral_norm(nn.Linear(in_dim, out_dim), n_power_iterations=5)
                # self.downsample = nn.Linear(in_dim, out_dim)
                self.bn = nn.BatchNorm1d(out_dim)
                init.kaiming_normal_(self.downsample.weight_orig, nonlinearity='leaky_relu')
            else:
                self.downsample = None

        def forward(self, x):
            out = self.swish(self.bn(self.fc(x)))

            if self.downsample is not None:
                x = self.downsample(x)
            return out + x

    def __init__(self, Encoder_layer_dims,hidden_dim_layer_out_Z, z_dim, class_num):
        super().__init__()
        # Encoder

        # Resblock
        self.Encoder_resblocks = nn.ModuleList()

        for i in range(len(Encoder_layer_dims) - 1):
            self.Encoder_resblocks.append(self.ResBlock(Encoder_layer_dims[i], Encoder_layer_dims[i + 1]))

        # 应用谱正则化和权重归一化
        self.fc21 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim), n_power_iterations=5)  # mu layer
        init.xavier_normal_(self.fc21.weight_orig)

        self.fc22 = spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim),
                                  n_power_iterations=5)  # logvariance layer
        init.xavier_normal_(self.fc22.weight_orig)


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
    

    



def modelevalution(model, loader):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            output, _ = model(data)
            all_predictions.extend(output.softmax(dim=1)[:, 1].tolist())  # probability for class 1
            all_targets.extend(target.tolist())
    return all_predictions, all_targets

def modelinference(model, loader):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for data in loader:
            output, _ = model(data[0])
            all_predictions.extend(output.softmax(dim=1)[:, 1].tolist())  # probability for class 1
    return all_predictions




def z_embedding(model, loader):
    model.eval()
    z_list = []  # 用来保存所有批次的 z 向量
    with torch.no_grad():  # 禁用梯度计算
        for data, _ in loader:
            _, z = model(data)  # 获取 class_logits 和 z
            z_list.append(z.cpu().numpy())  # 将每个批次的 z 转移到 CPU，并转换为 numpy 数组，添加到列表中
    # 将所有批次的 z 向量拼接成一个大的 numpy 数组
    z_numpy = np.concatenate(z_list, axis=0)
    return z_numpy










##input_data
filenames = os.listdir(open_path)
for filename in filenames:
    # scFoundation_Embedding = np.load(os.path.join(open_path, filename))
    full_path = os.path.join(open_path, filename)

    # 获取文件后缀名
    _, file_extension = os.path.splitext(filename)

    # 根据不同的文件后缀加载数据
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

# 检查数据是否含有NaN
print("DATA is containing NA?: ", np.isnan(scFoundation_Embedding).any())
print('Embedding data shape: ',scFoundation_Embedding.shape)
print('info data shape: ',scFoundation_Embedding_info.shape)


#读取预训练的panglao.npy,作为风格迁移的实验数据

full_path = os.path.join(open_path_conference_data, style_alignment_file)

bulk_1_10_Embedding = np.load(full_path)
print("Loaded .npy file:", bulk_1_10_Embedding)
print('style_alignment_file data shape: ',bulk_1_10_Embedding.shape)




##域适应迁移
def adain(input_data_Embedding, style_to_Embedding):
    """
    实现 AdaIN 的自适应实例归一化操作。

    scFoundation_Embedding: 需要标准化的特征 (m_samples, n_features)
    bulk_1_10_Embedding: 用于提供目标均值和方差的特征 (n_samples, n_features)
    """
    # 计算 scFoundation_Embedding 的均值和标准差 (按特征维度，即列)
    mu_x = np.mean(input_data_Embedding, axis=0, keepdims=True)  # 计算列均值
    sigma_x = np.std(input_data_Embedding, axis=0, keepdims=True)  # 计算列标准差

    # 计算 bulk_1_10_Embedding 的均值和标准差 (按特征维度，即列)
    mu_y = np.mean(style_to_Embedding, axis=0, keepdims=True)  # 计算列均值
    sigma_y = np.std(style_to_Embedding, axis=0, keepdims=True)  # 计算列标准差

    # 标准化 scFoundation_Embedding
    normalized_x = (input_data_Embedding - mu_x) / (sigma_x + 1e-8)  # 防止除零

    # 使用 bulk_1_10_Embedding 的均值和方差重新缩放和偏移
    adain_output = sigma_y * normalized_x + mu_y

    return adain_output

# 调用 AdaIN 函数
scFoundation_Embedding_adain = adain(scFoundation_Embedding, bulk_1_10_Embedding)


# 打印输出的形状，验证是否成功
print("AdaIN output shape:", scFoundation_Embedding_adain.shape)




# ---------------------------
# Prepare data for evaluation/inference based on inference_only flag
if not inference_only:
    scFoundation_Embedding_info[drug_label_choose] = scFoundation_Embedding_info[drug_label_choose].map(label_mapping)
    actual_class_num = scFoundation_Embedding_info[drug_label_choose].nunique()
    print("Number of classes (Actual):", actual_class_num)
    if actual_class_num != class_num:
        raise ValueError(f"Error: The actual number of classes {actual_class_num} does not match the expected {class_num}.")
    X_inference = scFoundation_Embedding_adain
    y_inference = scFoundation_Embedding_info[drug_label_choose]
    X_inference_tensor = torch.Tensor(X_inference).to(device)
    y_inference_tensor = torch.LongTensor(y_inference.values).to(device)
    inference_dataset = TensorDataset(X_inference_tensor, y_inference_tensor)
    print("Prepared dataset for evaluation (with labels).")
else:
    print("Inference-only mode: Labels will not be included.")
    X_inference = scFoundation_Embedding_adain
    X_inference_tensor = torch.Tensor(X_inference).to(device)
    inference_dataset = TensorDataset(X_inference_tensor)
    print("Prepared dataset for inference (without labels).")

inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# ---------------------------
# Load model and freeze encoder parameters
load_model = VAEclassification(Encoder_layer_dims, hidden_dim_layer_out_Z, z_dim, class_num).to(device)
load_model.load_state_dict(torch.load(os.path.join(open_path_conference_data, model_parameters_file), map_location=device))
for param in load_model.Encoder_resblocks.parameters():
    param.requires_grad = False
for param in load_model.fc21.parameters():
    param.requires_grad = False
for param in load_model.fc22.parameters():
    param.requires_grad = False

# ---------------------------
# Evaluation/Inference using the modified functions
if not inference_only:
    predictions, targets = modelevalution(load_model, inference_loader)
    targets = [int(x) for x in targets]
    predictions_prob = [float(x) for x in predictions]
    predictions_label = [1 if x > args.PP_threhold else 0 for x in predictions_prob]
    
    current_auc = roc_auc_score(targets, predictions_prob)
    current_mcc = matthews_corrcoef(targets, predictions_label)
    current_f1 = f1_score(targets, predictions_label)
    current_recall = recall_score(targets, predictions_label)
    current_accuracy = accuracy_score(targets, predictions_label)
    current_precision = precision_score(targets, predictions_label)
    
    auc_scores = [current_auc]
    mcc_scores = [current_mcc]
    f1_scores = [current_f1]
    recall_scores = [current_recall]
    accuracy_scores = [current_accuracy]
    precision_scores = [current_precision]
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(targets, predictions_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#1f77b4', lw=1.5, label=f'ROC (AUC = {current_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.savefig(os.path.join(save_path, f"{file_prefix}inference_auc.pdf"), dpi=1000, format='pdf')
    
    # Plot PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(targets, predictions_prob)
    ap_score = average_precision_score(targets, predictions_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='#1f77b4', lw=1.5, label=f'PR Curve (AP = {ap_score:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right", fontsize=10)
    plt.savefig(os.path.join(save_path, f"{file_prefix}inference_PR.pdf"), dpi=1000, format='pdf')
    
    # Save evaluation metrics to Excel
    results_df = pd.DataFrame({
        "AUC": auc_scores,
        "MCC": mcc_scores,
        "F1 Score": f1_scores,
        "Recall": recall_scores,
        "Accuracy": accuracy_scores,
        "Precision": precision_scores
    })
    results_df.to_excel(os.path.join(save_path, f"{file_prefix}evaluation_metrics.xlsx"), index=False)
    
    # Save predictions with probabilities to Excel
    value_to_label = {v: k for k, v in label_mapping.items()}
    mapped_targets = [value_to_label[x] for x in targets]
    mapped_predictions = [value_to_label[x] for x in predictions_label]
    prediction_df = pd.DataFrame({
        'True Label': mapped_targets,
        'Predicted Label': mapped_predictions,
        'Predicted Probability': predictions_prob
    })
    prediction_df.to_excel(os.path.join(save_path, f"{file_prefix}_True_and_inference_label_prob_results.xlsx"), index=False)
else:
    predictions = modelinference(load_model, inference_loader)
    predictions_prob = [float(x) for x in predictions]
    predictions_label = [1 if x > args.PP_threhold else 0 for x in predictions_prob]
    value_to_label = {v: k for k, v in label_mapping.items()}
    mapped_predictions = [value_to_label[x] for x in predictions_label]
    prediction_df = pd.DataFrame({
        'Predicted Label': mapped_predictions,
        'Predicted Probability': predictions_prob
    })
    prediction_df.to_excel(os.path.join(save_path, f'{file_prefix}_inference_label_prob_results.xlsx'), index=False)








