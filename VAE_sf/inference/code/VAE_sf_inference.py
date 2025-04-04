
import numpy as np
import pandas as pd
import os
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# from optuna.samplers import TPESampler
# import torch.optim as optim
# from optuna.pruners import MedianPruner
# import logging
import torch.nn.init as init
# from sklearn.model_selection import KFold
# from tqdm import tqdm
import argparse
from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import re
import json
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, accuracy_score, precision_score, roc_curve, roc_auc_score,auc,precision_recall_curve, average_precision_score
from itertools import cycle
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.nn.utils import weight_norm,spectral_norm
from scipy.stats import norm
# import importlib
# Create the parser
from scipy.stats import entropy
from scipy.stats import gaussian_kde
import random



parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")

# Add arguments
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--open_path_conference_data", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data', help="Path to open conference data.")
parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")
# parser.add_argument("--model_parameters_file_pretraining", type=str, default='model_parameters.pth', help="File name for storing model parameters.")
parser.add_argument("--style_alignment_file", type=str)

parser.add_argument("--model_parameters_file", type=str, default='model_parameters.pth', help="File name for storing model parameters.")
# parser.add_argument("--best_parameter_name", type=str, default='best_hyperparameters.xlsx', help="File name for best hyperparameters.")
# parser.add_argument("--execute_model_post_analysis", type=bool, default=True, help="Flag to execute model post-analysis.")
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--path_to_add', type=str, default='/home/luozeyu/desktop/big_model/',
                    help=' system path add for import model')

parser.add_argument('--epoch_start_for_loss_plot_only', type=int, default=1, help='Epoch start for loss plot only')
# parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
# parser.add_argument('--weight_decay', type=float, default=1e-3, help='l2 norm weight')
# parser.add_argument('--num_epochs', type=int, default=50, help='num_epochs for training')
parser.add_argument('--device_choose', type=str, default='cuda:0', help='cpu choose')
parser.add_argument('--label_mapping', type=str, default='{"tos": 0, "toR": 1}',
                    help='JSON string for label mapping e.g. \'{"tos": 0, "toR": 1}\'')
parser.add_argument('--class_num', type=int, default=2, help='Number of classes.')
parser.add_argument('--drug_label_choose', type=str, default='label', help='label of classes.')
parser.add_argument('--seed_set', type=int, default=42, help='Number of random seed.')


# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
open_path = args.open_path
save_path = args.save_path
open_path_conference_data = args.open_path_conference_data
file_prefix = args.file_prefix
model_parameters_file = args.model_parameters_file
# model_parameters_file_pretraining = args.model_parameters_file_pretraining
# best_parameter_name = args.best_parameter_name
# execute_model_post_analysis = args.execute_model_post_analysis
batch_size = args.batch_size
path_to_add = args.path_to_add
# learning_rate = args.learning_rate
# weight_decay = args.weight_decay
# num_epochs = args.num_epochs
epoch_start_for_loss_plot_only = args.epoch_start_for_loss_plot_only
device_choose = args.device_choose
style_alignment_file = args.style_alignment_file


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
# print(f"Best Parameter Name: {best_parameter_name}")
# print(f"Execute Model Post Analysis: {execute_model_post_analysis}")
print(f"Epoch Start for Loss Plot Only: {epoch_start_for_loss_plot_only}")
print(f'path_to_add:{path_to_add}')
print("Label Mapping:", label_mapping)
print("Number of Classes (Expected):", class_num)
# 检查目录是否存在，如果不存在则创建
if not os.path.exists(save_path):
    os.makedirs(save_path)

##############################添加模型定义所在目录到系统路径, 实现vae_pretraning encoder调用
if path_to_add not in sys.path:
    sys.path.append(path_to_add)


from VAE_sf_pretraining_model.config import config

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

    def __init__(self, Encoder_layer_dims,hidden_dim_layer_out_Z, z_dim, class_num):
        super(VAEclassification, self).__init__()
        # Encoder

        # Resblock
        self.Encoder_resblocks = nn.ModuleList()

        for i in range(len(Encoder_layer_dims) - 1):
            self.Encoder_resblocks.append(self.ResBlock(Encoder_layer_dims[i], Encoder_layer_dims[i + 1]))

        # Latent space
        self.fc21 = nn.Linear(hidden_dim_layer_out_Z, z_dim)  # mu layer
        init.xavier_normal_(self.fc21.weight)  # Xavier Initialization for mu layer
        self.fc22 = nn.Linear(hidden_dim_layer_out_Z, z_dim)  # logvariance layer
        init.xavier_normal_(self.fc22.weight)  # Xavier Initialization for logvar layer

        self.fc3 = spectral_norm(nn.Linear(z_dim, z_dim // 2), n_power_iterations=5)
        init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        self.bn3 = nn.BatchNorm1d(z_dim // 2)
        self.swish = swish
        self.fc4 = spectral_norm(nn.Linear(z_dim // 2, class_num), n_power_iterations=5)
        init.xavier_normal_(self.fc4.weight)

        # classfication_MLP
        # self.classifier = nn.Sequential(
        #     nn.Linear(z_dim, z_dim//2),
        #     nn.BatchNorm1d(z_dim//2),
        #     nn.LeakyReLU(),
        #     nn.Linear(z_dim//2, class_num),
        #
        #     spectral_norm(nn.Linear(hidden_dim_layer_out_Z, z_dim), n_power_iterations=3), # mu layer
        #     init.xavier_normal_(self.fc21.weight)
        # )
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




def modelinference(model, loader):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            output,_ = model(data)
            all_predictions.extend(output.softmax(dim=1)[:, 1].tolist())  # 针对二分类任务的第二类的概率
            all_targets.extend(target.tolist())
    return all_predictions, all_targets



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


def plot_confusion_matrix(targets, predictions_label, class_labels, save_path, file_name,
                          figsize=(10, 8), percentage=False, cmap='viridis_r', fmt=None):
    """
    通用绘制混淆矩阵的函数，可以选择绘制绝对数值或百分比。

    Parameters
    ----------
    targets : list or np.array
        真实标签。
    predictions_label : list or np.array
        预测标签。
    class_labels : list or np.array
        类别标签，顺序应与模型的默认分类顺序一致。
    save_path : str
        文件保存路径（不包括文件名）。
    file_name : str
        文件名，不包括扩展名。
    figsize : tuple, optional
        图像大小，默认值为 (10, 8)。
    percentage : bool, optional
        是否绘制百分比的混淆矩阵，默认值为 False。
    cmap : str, optional
        颜色图谱，默认值为 'viridis_r'。
    fmt : str, optional
        数值格式，默认为 None。对于百分比可设置为 '.2%'

    Returns
    -------
    None
    """
    # 生成混淆矩阵
    matrix = confusion_matrix(targets, predictions_label, labels=class_labels)

    if percentage:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2%' if fmt is None else fmt
        title_suffix = "(Percentage)"
    else:
        fmt = 'd' if fmt is None else fmt
        title_suffix = "(Absolute)"

    # 设置标签
    xticklabels = [str(label) for label in class_labels]
    yticklabels = [str(label) for label in class_labels]

    # 设置图形大小
    plt.figure(figsize=figsize)

    # 使用 seaborn 绘制热图
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap, xticklabels=xticklabels, yticklabels=yticklabels)

    # 配置图形细节
    plt.gca().invert_yaxis()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'{file_name} Confusion Matrix {title_suffix}')

    # 保存图像
    plt.savefig(os.path.join(save_path,f"{file_name}_confusion_matrix_{title_suffix.strip('()').lower()}.png"), dpi=1000,
                bbox_inches="tight")
    plt.savefig(os.path.join(save_path,f"{file_name}_confusion_matrix_{title_suffix.strip('()').lower()}.pdf"), dpi=1000,
                bbox_inches="tight")
    plt.close()


def parse_label_mapping(label_mapping):

    # 按数值索引排序类别名称
    sorted_labels = sorted(label_mapping.items(), key=lambda item: item[1])
    class_labels = [item[0] for item in sorted_labels]

    return class_labels




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




# 定义计算KL散度的函数
def compute_kl_divergence(p, q):
    """
    计算两个概率分布之间的KL散度。
    """
    # 添加一个小的值以避免零概率
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    return entropy(p, q)


def get_probability_distribution(data, bins):
    """
    计算数据的概率分布（直方图归一化）。
    """
    hist, _ = np.histogram(data, bins=bins, density=True)
    # 计算每个bin的面积
    bin_width = bins[1] - bins[0]
    hist = hist * bin_width
    # 归一化
    hist = hist / np.sum(hist)
    return hist


def calculate_kl_per_feature(data1, data2, num_bins=30):
    """
    计算每个特征维度的数据1和数据2之间的KL散度。
    """
    n_features = data1.shape[1]
    kl_divergences = []
    for feature in range(n_features):
        # 获取两个数据集在该特征的所有样本
        feature_data1 = data1[:, feature]
        feature_data2 = data2[:, feature]

        # 确定共同的bin边界
        combined_data = np.concatenate([feature_data1, feature_data2])
        bins = np.histogram_bin_edges(combined_data, bins=num_bins)

        # 计算概率分布
        p = get_probability_distribution(feature_data1, bins)
        q = get_probability_distribution(feature_data2, bins)

        # 计算KL散度
        kl = compute_kl_divergence(p, q)
        kl_divergences.append(kl)

    return np.array(kl_divergences)


# 定义计算分布范围差异度量的函数
def compute_distribution_overlap(data1, data2, bins=30):
    """
    计算两个数据集的分布重叠度。
    """
    # 计算两个数据的概率密度函数
    kde1 = gaussian_kde(data1)
    kde2 = gaussian_kde(data2)

    # 创建一个值的范围来计算密度
    min_value = min(data1.min(), data2.min())
    max_value = max(data1.max(), data2.max())
    x_values = np.linspace(min_value, max_value, bins)

    # 计算两个分布的密度
    density1 = kde1(x_values)
    density2 = kde2(x_values)

    # 计算分布的重叠度
    overlap = np.sum(np.minimum(density1, density2)) * (x_values[1] - x_values[0])

    return overlap



# 定义可视化并保存图像的函数
def plot_feature_distributions(data1, data2, feature_indices, kl_divs, category, save_path, file_prefix,
                               dataset1_label='scFoundation_Embedding', dataset2_label='bulk_1_10_Embedding'):
    """
    绘制选定特征维度的直方图对比，并保存图像。
    """
    for feature in feature_indices:
        plt.figure(figsize=(8, 6))
        sns.histplot(data1[:, feature], bins=30, color='blue', alpha=0.5, label=dataset1_label, stat='density')
        sns.histplot(data2[:, feature], bins=30, color='red', alpha=0.5, label=dataset2_label, stat='density')
        plt.title(f'Feature {feature} Distribution Comparison\nCategory: {category}, KL: {kl_divs[feature]:.4f}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()

        # 保存图像到指定子文件夹
        if category == 'Top 20':
            subfolder = 'top_20'
        else:
            subfolder = 'bottom_20'

        # 创建子文件夹
        subfolder_path = os.path.join(save_path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # 保存图像
        plot_filename = f'{file_prefix}_feature_{feature}_{subfolder}.pdf'
        plot_filepath = os.path.join(subfolder_path, plot_filename)
        plt.savefig(plot_filepath, dpi=1000)
        plt.close()
        print(f"Saved plot for feature {feature} to {plot_filepath}")



# 计算每个特征的KL散度
kl_divs = calculate_kl_per_feature(scFoundation_Embedding, bulk_1_10_Embedding, num_bins=30)

# 创建一个DataFrame来存储KL散度结果
df_kl = pd.DataFrame({
    'Feature_Index': np.arange(scFoundation_Embedding.shape[1]),
    'KL_Divergence': kl_divs
})

# 计算KL散度的均值
kl_mean = np.mean(kl_divs)


# 创建一个包含均值的新行的 DataFrame
df_kl_mean = pd.DataFrame({'Feature_Index': ['Mean'], 'KL_Divergence': [kl_mean]})

# 使用 pd.concat() 将均值行与原来的 DataFrame 合并
df_kl = pd.concat([df_kl, df_kl_mean], ignore_index=True)

# 保存KL散度结果到Excel文件
excel_path = os.path.join(save_path, f'{file_prefix}_KL_divergence.xlsx')
df_kl.to_excel(excel_path, index=False)
print(f"KL divergence results saved to {excel_path}")


# 按照KL散度降序排序
df_kl_sorted = df_kl.sort_values(by='KL_Divergence', ascending=False)

# 选择KL散度最大的20个特征和最小的20个特征
top_20 = df_kl_sorted.head(20).reset_index(drop=True)
bottom_20 = df_kl_sorted.tail(20).reset_index(drop=True)

# 添加类别标签
top_20['Category'] = 'Top 20'
bottom_20['Category'] = 'Bottom 20'

# 合并Top 20和Bottom 20
df_selected = pd.concat([top_20, bottom_20], ignore_index=True)

# 获取Top 20和Bottom 20的特征索引
top_20_indices = top_20['Feature_Index'].values
bottom_20_indices = bottom_20['Feature_Index'].values

# 绘制并保存KL散度最大的20个特征的分布图
print("Saving plots for top 20 features with highest KL divergence...")
plot_feature_distributions(
    scFoundation_Embedding,
    bulk_1_10_Embedding,
    top_20_indices,
    kl_divs,
    category='Top 20',
    save_path=save_path,
    file_prefix=file_prefix
)

# 绘制并保存KL散度最小的20个特征的分布图
print("Saving plots for bottom 20 features with lowest KL divergence...")
plot_feature_distributions(
    scFoundation_Embedding,
    bulk_1_10_Embedding,
    bottom_20_indices,
    kl_divs,
    category='Bottom 20',
    save_path=save_path,
    file_prefix=file_prefix
)



print("All Top 10 combined KL and overlap divergence plots have been saved.")

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
##同理查看adain后的分布变化风格

# 计算每个特征的KL散度
kl_divs = calculate_kl_per_feature(scFoundation_Embedding_adain, bulk_1_10_Embedding, num_bins=30)

# 创建一个DataFrame来存储KL散度结果
df_kl = pd.DataFrame({
    'Feature_Index': np.arange(scFoundation_Embedding_adain.shape[1]),
    'KL_Divergence': kl_divs
})

# 计算KL散度的均值
kl_mean = np.mean(kl_divs)

# 创建一个包含均值的新行的 DataFrame
df_kl_mean = pd.DataFrame({'Feature_Index': ['Mean'], 'KL_Divergence': [kl_mean]})

# 使用 pd.concat() 将均值行与原来的 DataFrame 合并
df_kl = pd.concat([df_kl, df_kl_mean], ignore_index=True)

# 保存KL散度结果到Excel文件
excel_path = os.path.join(save_path, f'{file_prefix}_adain_KL_divergence.xlsx')
df_kl.to_excel(excel_path, index=False)


# 按照KL散度降序排序
df_kl_sorted = df_kl.sort_values(by='KL_Divergence', ascending=False)

# 选择KL散度最大的20个特征和最小的20个特征
top_20 = df_kl_sorted.head(20).reset_index(drop=True)
bottom_20 = df_kl_sorted.tail(20).reset_index(drop=True)

# 添加类别标签
top_20['Category'] = 'Top 20'
bottom_20['Category'] = 'Bottom 20'

# 合并Top 20和Bottom 20
df_selected = pd.concat([top_20, bottom_20], ignore_index=True)

# 获取Top 20和Bottom 20的特征索引
top_20_indices = top_20['Feature_Index'].values
bottom_20_indices = bottom_20['Feature_Index'].values

# 绘制并保存KL散度最大的20个特征的分布图
print("Saving plots for top 20 features with highest KL divergence...")
plot_feature_distributions(
    scFoundation_Embedding_adain,
    bulk_1_10_Embedding,
    top_20_indices,
    kl_divs,
    category='Top 20',
    save_path=save_path+'/after_adain',
    file_prefix=file_prefix
)

# 绘制并保存KL散度最小的20个特征的分布图
print("Saving plots for bottom 20 features with lowest KL divergence...")
plot_feature_distributions(
    scFoundation_Embedding_adain,
    bulk_1_10_Embedding,
    bottom_20_indices,
    kl_divs,
    category='Bottom 20',
    save_path=save_path+'/after_adain',
    file_prefix=file_prefix
)




















# 启用异常检测
torch.autograd.set_detect_anomaly(True)


# 初始化5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 用于存储每一折的评分
scores = []

# 假设 scFoundation_Embedding_info 是一个DataFrame
scFoundation_Embedding_info[drug_label_choose] = scFoundation_Embedding_info[drug_label_choose].map(label_mapping)

# 计算实际类别数
actual_class_num = scFoundation_Embedding_info[drug_label_choose].nunique()
print("Number of classes (Actual):", actual_class_num)

# 比较预期类别数与实际类别数
if actual_class_num != class_num:
    raise ValueError(f"Error: The actual number of classes {actual_class_num} does not match the expected {class_num}.")


# 初始化存储这些指标的列表
auc_scores = []
mcc_scores = []
f1_scores = []
recall_scores = []
accuracy_scores = []
precision_scores = []


# 分割数据
X_inference = scFoundation_Embedding_adain
y_inference = scFoundation_Embedding_info[drug_label_choose]

X_inference_tensor = torch.Tensor(X_inference).to(device)
y_inference_tensor = torch.LongTensor(y_inference.values).to(device)


# 创建数据加载器
inference_dataset = TensorDataset(X_inference_tensor, y_inference_tensor)
inference_loader = DataLoader(inference_dataset,batch_size=batch_size, shuffle=False,
                        drop_last=False)



##define model

load_model = VAEclassification(Encoder_layer_dims,hidden_dim_layer_out_Z, z_dim, class_num).to(device)
load_model.load_state_dict(torch.load(os.path.join(open_path_conference_data, model_parameters_file), map_location=device))

#固定预训练Encoder的权重
for param in load_model.Encoder_resblocks.parameters():
    param.requires_grad = False
for param in load_model.fc21.parameters():
    param.requires_grad = False
for param in load_model.fc22.parameters():
    param.requires_grad = False



predictions, targets = modelinference(load_model, inference_loader)


# 计算各种指标
targets = [int(x) for x in targets]
predictions_prob = [float(x) for x in predictions]
predictions_label = [1 if x > 0.5 else 0 for x in predictions_prob]


current_auc = roc_auc_score(targets, predictions_prob)
# print(current_auc)
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

# 为ROC曲线存储FPR和TPR
fpr, tpr, _ = roc_curve(targets, predictions_prob)


# 为PR曲线存储precision和recall
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

# 保存图像到文件
plt.savefig(os.path.join(save_path,f"{file_prefix}inference_auc.pdf"),dpi = 1000, format='pdf')



results_df = pd.DataFrame({
    "AUC": auc_scores,
    "MCC": mcc_scores,
    "F1 Score": f1_scores,
    "Recall": recall_scores,
    "Accuracy": accuracy_scores,
    "Precision": precision_scores
})

# 保存DataFrame到Excel
results_df.to_excel(os.path.join(save_path, f"{file_prefix}evaluation_metrics.xlsx"), index=False)


##PR曲线绘制
# 假设predictions_prob_list和targets_list包含了多个折的数据

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
# 保存图像到文件
plt.savefig(os.path.join(save_path,f"{file_prefix}inference_PR.pdf"),dpi = 1000, format='pdf')


z_numpy = z_embedding(load_model, inference_loader)

# 保存为 .npy 文件
np.save(os.path.join(save_path, f"{file_prefix}latentz_embedding.npy"), z_numpy)  # 保存为 .npy 文件



def z_latent_plot(latent_vectors,save_path,file_prefix):
    ## each Z feature Distribution plot and Normality Testing

    n_cols = 10  # 每行展示5个图表
    n_rows = (z_dim + n_cols - 1) // n_cols  # 计算需要多少行

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))  # 调整图表大小
    axes = axes.flatten()  # 展平axes数组以便索引

    for i in range(z_dim):
        data = latent_vectors[:, i]
        ax = axes[i]
        ax.hist(data, bins=30, density=True, alpha=0.6, color='g')

        mu, std = norm.fit(data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2)
        title = f"mu = {mu:.2f},std = {std:.2f},\nDim{i}"
        ax.set_title(title)

    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f"{file_prefix}latent_space_distribution.pdf"), dpi=1000)  # 保存整个图表

def Kl_Statistics(latent_vectors,save_path,file_prefix):
    results = []

    for i in range(z_dim):
        data = latent_vectors[:, i]
        # 计算数据的直方图作为概率分布
        data_hist, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 生成与数据直方图相同bins的正态分布概率密度
        norm_pdf = norm.pdf(bin_centers, loc=np.mean(data), scale=np.std(data))

        # 计算KL散度
        kl_divergence = entropy(data_hist, norm_pdf)

        result_dict = {
            "Dimension": i,
            "KL Divergence": kl_divergence
        }
        results.append(result_dict)

    # 计算KL散度的均值和方差
    kl_values = [result['KL Divergence'] for result in results]
    average_kl = np.mean(kl_values)
    variance_kl = np.var(kl_values)

    # 添加平均KL散度和方差到DataFrame
    results_df = pd.DataFrame(results)
    results_df.loc['Average', 'KL Divergence'] = average_kl
    results_df.loc['Variance', 'KL Divergence'] = variance_kl

    # 保存DataFrame到Excel文件
    results_df.to_excel(os.path.join(save_path, f"{file_prefix}kl_divergence_analysis.xlsx"))


z_latent_plot(z_numpy,save_path,file_prefix)

Kl_Statistics(z_numpy,save_path,file_prefix)



# 使用通用绘制函数绘制绝对数值混淆矩阵 confusion matrix


class_labels = parse_label_mapping(label_mapping)
print("Class Labels:", class_labels)

value_to_label = {v: k for k, v in label_mapping.items()}

# 映射真实标签和预测标签
mapped_targets = [value_to_label[x] for x in targets]
mapped_predictions = [value_to_label[x] for x in predictions_label]

plot_confusion_matrix(
    targets=mapped_targets,
    predictions_label=mapped_predictions,
    class_labels=class_labels,
    save_path=save_path,
    file_name=file_prefix,
    figsize=(10, 8),
    percentage=False,
    cmap='viridis_r',
    fmt='d'
)

# 使用通用绘制函数绘制百分比混淆矩阵
plot_confusion_matrix(
    targets=mapped_targets,
    predictions_label=mapped_predictions,
    class_labels=class_labels,
    save_path=save_path,
    file_name=file_prefix,
    figsize=(10, 8),
    percentage=True,
    cmap='viridis_r',
    fmt='.2%'
)

##保存推理的数据预测标签及预测概率
# 创建 DataFrame
prediction_df = pd.DataFrame({
    # 'True Label': targets,
    'Predicted Label': mapped_predictions,
    'Predicted Probability': predictions_prob
})

# 保存到 Excel 文件
prediction_df.to_excel(os.path.join(save_path, f'{file_prefix}_inference_label_prob_results.xlsx'), index=False ) # 确保不保存行索引