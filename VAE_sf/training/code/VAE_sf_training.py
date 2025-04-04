
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from optuna.samplers import TPESampler
import torch.optim as optim
from optuna.pruners import MedianPruner
import logging
import torch.nn.init as init
from sklearn.model_selection import KFold
from tqdm import tqdm
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import re
import json
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score, accuracy_score, precision_score, roc_curve, roc_auc_score,auc,precision_recall_curve, average_precision_score
from itertools import cycle
from torch.nn.utils import weight_norm,spectral_norm
from scipy.stats import entropy
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import TomekLinks
import random
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline




def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')


# Create the parser
parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")

# Add arguments
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--open_path_conference_data", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data', help="Path to open conference data.")
parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")
parser.add_argument("--model_parameters_file_pretraining", type=str, default='model_parameters.pth', help="File name for storing model parameters.")
parser.add_argument("--style_alignment_file", type=str)

parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--model_configuration_path', type=str, default='/home/luozeyu/desktop/big_model/',
                    help=' system path add for import model')

parser.add_argument('--epoch_start_for_loss_plot_only', type=int, default=1, help='Epoch start for loss plot only')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='l2 norm weight')
parser.add_argument('--num_epochs', type=int, default=50, help='num_epochs for training')
parser.add_argument('--device_choose', type=str, default='cuda:0', help='cpu choose')
parser.add_argument('--label_mapping', type=str, default='{"tos": 0, "toR": 1}',
                    help='JSON string for label mapping e.g. \'{"tos": 0, "toR": 1}\'')
parser.add_argument('--class_num', type=int, default=2, help='Number of classes.')

parser.add_argument('--drug_label_choose', type=str, default='label', help='label of classes.')
parser.add_argument("--VAE_augmentation_used", type=strict_str2bool, default=False, help="if synthetic minority over-sampling based on VAE.")

parser.add_argument('--multiplier_choose', type=str, choices=['auto', 'from_presetting'], default='from_presetting',
                    help='Choose "auto" to automatically adjust multipliers or "from_presetting" to use preset values')
parser.add_argument('--minority_multiplier', type=float, default=1.5,
                        help='Multiplier for the minority class (default: 1.5)')
parser.add_argument('--majority_multiplier', type=float, default=1.0,
                        help='Multiplier for the majority class (default: 1.0)')
parser.add_argument("--SMOTE_used", type=strict_str2bool, default=False, help="if synthetic minority over-sampling.")
parser.add_argument("--post_training", type=strict_str2bool, default=False, help=" ")
parser.add_argument('--seed_set', type=int, default=42, help='Number of random seed.')
parser.add_argument("--post_training_epoch_num", type=int, default=10, help=" ")

# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
open_path = args.open_path
save_path = args.save_path
open_path_conference_data = args.open_path_conference_data
file_prefix = args.file_prefix
model_parameters_file_pretraining = args.model_parameters_file_pretraining
style_alignment_file = args.style_alignment_file

batch_size = args.batch_size
path_to_add = args.model_configuration_path
learning_rate = args.learning_rate
weight_decay = args.weight_decay
num_epochs = args.num_epochs
epoch_start_for_loss_plot_only = args.epoch_start_for_loss_plot_only
device_choose = args.device_choose

label_mapping = json.loads(args.label_mapping)
class_num = args.class_num
drug_label_choose  = args.drug_label_choose
VAE_augmentation_used = args.VAE_augmentation_used

minority_multiplier =args.minority_multiplier
majority_multiplier =args.majority_multiplier

SMOTE_used = args.SMOTE_used
post_training  = args.post_training
post_training_epoch_num = args.post_training_epoch_num
seed_set = args.seed_set

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(seed_set)


if SMOTE_used and VAE_augmentation_used:
    raise ValueError("SMOTE and VAE augmentation should not be used together")
elif SMOTE_used:
    print("SMOTE is used, but not VAE augmentation.")
elif VAE_augmentation_used:
    print("VAE augmentation is used, but not SMOTE.")
else:
    print("No augmentation is used.")


# REC_beta = args.REC_beta

# Now you can use these variables in your script
print(f"Open Path: {open_path}")
print(f"Save Path: {save_path}")
# print(f"Open Path Conference Data: {open_path_conference_data}")
print(f"File Prefix: {file_prefix}")
print(f"Epoch Start for Loss Plot Only: {epoch_start_for_loss_plot_only}")
print(f'path_to_add:{path_to_add}')
print("Label Mapping:", label_mapping)
print("Number of Classes (Expected):", class_num)
print("is VAE_augmentation_used:", VAE_augmentation_used)


if not os.path.exists(save_path):
    os.makedirs(save_path)

##############################添加模型定义所在目录到系统路径, 实现vae_pretraning encoder调用
if path_to_add not in sys.path:
    sys.path.append(path_to_add)

from VAE_sf_pretraining_model.RecResVAE import ContinuousResidualVAE
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
#
# model_pretraining = ContinuousResidualVAE(input_dim=input_dim, hidden_dim_layer0=hidden_dim_layer0,
#                                   Encoder_layer_dims=Encoder_layer_dims, Decoder_layer_dims=Decoder_layer_dims,
#                                   hidden_dim_layer_out_Z=hidden_dim_layer_out_Z, z_dim=z_dim, loss_type='MSE',
#                                   reduction='mean').to(device)
#
# model_pretraining.load_state_dict(torch.load(os.path.join(path_to_add,model_parameters_file_pretraining), map_location=device))




def freeze_layers(model):
    # 冻结特定层的参数，并将其设置为评估模式
    for param in model.Encoder_resblocks.parameters():
        param.requires_grad = False
    for param in model.fc21.parameters():
        param.requires_grad = False
    for param in model.fc22.parameters():
        param.requires_grad = False
    model.Encoder_resblocks.eval()
    model.fc21.eval()
    model.fc22.eval()

def unfreeze_layers(model):
    # 解冻特定层的参数，并将其设置为训练模式
    for param in model.Encoder_resblocks.parameters():
        param.requires_grad = True
    for param in model.fc21.parameters():
        param.requires_grad = True
    for param in model.fc22.parameters():
        param.requires_grad = True
    model.Encoder_resblocks.train()
    model.fc21.train()
    model.fc22.train()

# 在训练过程中调用
# # 初始时冻结
# freeze_layers(model_pretraining)

#
# # 初始阶段，冻结指定层的权重
# set_requires_grad(model_pretraining, ['Encoder_resblocks', 'fc21', 'fc22'], requires_grad=False)

############################################


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
    def __init__(self, model_pretraining, z_dim, class_num):
        super(VAEclassification, self).__init__()
        self.Encoder_resblocks  = model_pretraining.Encoder_resblocks
        self.fc21 = model_pretraining.fc21
        self.fc22 = model_pretraining.fc22

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
        return class_logits



# 训练函数
def train(model, train_loader,optimizer,criterion):

    model.train()  # 设置模型为训练模式

    total_loss = 0  # 累计所有批次的损失

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)  # Calculate the loss using both outputs and labels

        # 反向传播和优化
        loss.backward()
        # clip_grad_norm_(model.parameters(), max_norm=1.0)  # 将梯度的范数裁剪到1.0
        optimizer.step()


        # 累计损失
        total_loss += loss
    # 计算并返回平均损失
    avg_loss = total_loss / len(train_loader)

    return avg_loss


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(loader)



def validate_ACC_final_epoch(model, loader):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            all_predictions.extend(output.softmax(dim=1)[:, 1].tolist())  # 针对二分类任务的第二类的概率
            all_targets.extend(target.tolist())
    return all_predictions, all_targets





def smote_tomek_preprocess(X_train, y_train, random_state=42):
    # 计算类别样本数量
    counter = Counter(y_train)
    print("原始类别分布：", counter)

    # 创建采样器的 Pipeline
    pipeline = Pipeline([
        ('oversample', SMOTE(random_state=random_state)),
        ('undersample', TomekLinks())
        #('undersample', NeighbourhoodCleaningRule())
    ])

    # 对训练集进行采样
    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    print("处理后类别分布：", Counter(y_resampled))

    return X_resampled, y_resampled






def augment_data_with_decoder(model, X_train, y_train, minority_multiplier, majority_multiplier):
    model.eval()  # 设置模型为评估模式

    X_train_tensor = torch.Tensor(X_train).to(device)
    # y_train_tensor = torch.LongTensor(y_train).to(device)

    X_augmented = []
    y_augmented = []
    # 假设 y_train 是包含类别标签的 NumPy 数组
    class_counts = Counter(y_train)
    # 确定少数类和多数类
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)

    print(f"少数类: {minority_class}, 多数类: {majority_class}")

    with torch.no_grad():
        # 对每个类别进行处理
        for cls in np.unique(y_train):
            cls_indices = np.where(y_train == cls)[0]
            X_cls = X_train_tensor[cls_indices]
            n_samples_cls = len(X_cls)
            print('X_cls个数：' ,n_samples_cls)

            # 根据类别选择适当的乘数
            if cls == minority_class:
                multiplier = minority_multiplier
            elif cls == majority_class:
                multiplier = majority_multiplier
            print(multiplier)

            n_new_samples = int((multiplier - 1) * n_samples_cls)
            print('n_new_samples个数：',n_new_samples)
            if n_new_samples <= 0:
                continue

            # 获取潜在向量的均值和方差
            mu, logvar = model.encode(X_cls)
            # z_dim = mu.size(1)

            # 生成新的 z 向量
            z_new_samples = []
            for _ in range(n_new_samples):
                idx = np.random.randint(0, n_samples_cls)
                z_sample = model.reparameterize(mu[idx], logvar[idx])
                z_new_samples.append(z_sample.unsqueeze(0))
            z_new_samples = torch.cat(z_new_samples, dim=0)

            # 解码生成新样本
            X_new_samples = model.decode(z_new_samples)

            X_augmented.append(X_new_samples)
            y_augmented.append(torch.full((n_new_samples,), cls, dtype=torch.long).to(device))

    if X_augmented:
        X_augmented = torch.cat(X_augmented, dim=0)
        y_augmented = torch.cat(y_augmented, dim=0)
    else:
        X_augmented = torch.Tensor([]).to(device)
        y_augmented = torch.LongTensor([]).to(device)

    return X_augmented, y_augmented





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

##读取预训练的panglao.npy,作为风格迁移的实验数据

full_path = os.path.join(open_path_conference_data, style_alignment_file)

scpanglao_1_10_Embedding = np.load(full_path)
print("Loaded .npy file:", scpanglao_1_10_Embedding)
print('style_alignment_file data shape: ',scpanglao_1_10_Embedding.shape)


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
                               dataset1_label='scFoundation_Embedding', dataset2_label='scpanglao_1_10_Embedding'):
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
kl_divs = calculate_kl_per_feature(scFoundation_Embedding, scpanglao_1_10_Embedding, num_bins=30)

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
    scpanglao_1_10_Embedding,
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
    scpanglao_1_10_Embedding,
    bottom_20_indices,
    kl_divs,
    category='Bottom 20',
    save_path=save_path,
    file_prefix=file_prefix
)



print("All Top 10 combined KL and overlap divergence plots have been saved.")

##域适应迁移
def adain(bulk_Embedding, scRNAseqEmbedding):
    """
    实现 AdaIN 的自适应实例归一化操作。

    scFoundation_Embedding: 需要标准化的特征 (m_samples, n_features)
    scpanglao_1_10_Embedding: 用于提供目标均值和方差的特征 (n_samples, n_features)
    """
    # 计算 scFoundation_Embedding 的均值和标准差 (按特征维度，即列)
    mu_x = np.mean(bulk_Embedding, axis=0, keepdims=True)  # 计算列均值
    sigma_x = np.std(bulk_Embedding, axis=0, keepdims=True)  # 计算列标准差

    # 计算 scpanglao_1_10_Embedding 的均值和标准差 (按特征维度，即列)
    mu_y = np.mean(scRNAseqEmbedding, axis=0, keepdims=True)  # 计算列均值
    sigma_y = np.std(scRNAseqEmbedding, axis=0, keepdims=True)  # 计算列标准差

    # 标准化 scFoundation_Embedding
    normalized_x = (bulk_Embedding - mu_x) / (sigma_x + 1e-8)  # 防止除零

    # 使用 scpanglao_1_10_Embedding 的均值和方差重新缩放和偏移
    adain_output = sigma_y * normalized_x + mu_y

    return adain_output

# 调用 AdaIN 函数
scFoundation_Embedding_adain = adain(scFoundation_Embedding, scpanglao_1_10_Embedding)


# 打印输出的形状，验证是否成功
print("AdaIN output shape:", scFoundation_Embedding_adain.shape)
##同理查看adain后的分布变化风格

# 计算每个特征的KL散度
kl_divs = calculate_kl_per_feature(scFoundation_Embedding_adain, scpanglao_1_10_Embedding, num_bins=30)

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
    scpanglao_1_10_Embedding,
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
    scpanglao_1_10_Embedding,
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



# 假设 scFoundation_Embedding_info 是一个DataFrame
scFoundation_Embedding_info[drug_label_choose] = scFoundation_Embedding_info[drug_label_choose].map(label_mapping)
# 计算并显示每个标签的数量
label_counts = scFoundation_Embedding_info[drug_label_choose].value_counts()
print(f"Counts of each label in {drug_label_choose}:\n{label_counts}")

# 计算实际类别数
actual_class_num = scFoundation_Embedding_info[drug_label_choose].nunique()
print("Number of classes (Actual):", actual_class_num)

# 比较预期类别数与实际类别数
if actual_class_num != class_num:
    raise ValueError(f"Error: The actual number of classes {actual_class_num} does not match the expected {class_num}.")


#重新设定--majority_multiplier和--minority_multiplier的数值，如果--multiplier_choose =auto
# Identify majority and minority classes

# 重新设定--majority_multiplier和--minority_multiplier的数值，如果--multiplier_choose =auto
# Identify majority and minority classes
def minmaxclass_auto_count(label_counts, args, save_path, file_prefix):
    if label_counts.values.tolist()[0] >= label_counts.values.tolist()[1]:
        # majority_class = label_counts.index.tolist()[0]
        # minority_class = label_counts.index.tolist()[1]
        majority_count = label_counts.values.tolist()[0]
        minority_count = label_counts.values.tolist()[1]
    else:
        # majority_class = label_counts.index.tolist()[1]
        # minority_class = label_counts.index.tolist()[0]
        majority_count = label_counts.values.tolist()[1]
        minority_count = label_counts.values.tolist()[0]

    if (majority_count - minority_count) < 0.3 * majority_count:
        print("Minor class count is close to major class count. Aligning without increasing major class.")
        majority_multiplier = 1
        minority_multiplier = majority_count / minority_count
        # Recompute adjusted majority count
        adjusted_majority_count = majority_multiplier * majority_count
        # Compute adjusted minority count
        adjusted_minority_count = minority_multiplier * minority_count

    elif args.multiplier_choose == 'auto':
        print("Auto mode selected. Ignoring preset multipliers and adjusting automatically.")

        # if args.multiplier_choose == 'auto':
        #     print("Auto mode selected. Ignoring preset multipliers and adjusting automatically.")

        # **Rule 1:** Multiply majority_multiplier by 1.5
        majority_multiplier = 1.0 * 1.5  # Starting from 1.0 as per default

        # Compute adjusted majority count
        adjusted_majority_count = majority_multiplier * majority_count

        # Attempt to set minority_multiplier so that adjusted counts are equal
        desired_minority_multiplier = adjusted_majority_count / minority_count

        # **Rule 2:** Limit minority_multiplier between 1.2 and 3
        minority_multiplier = min(max(desired_minority_multiplier, 1.2), 4)

        # Compute adjusted minority count
        adjusted_minority_count = minority_multiplier * minority_count

        # **Rule 3:** Check if adjusted minority count is less than half of adjusted majority count
        if adjusted_minority_count < 0.5 * adjusted_majority_count:
            # Limit majority_multiplier between 1 and 1.1
            majority_multiplier = 1  # Reset to original before limiting

            # Recompute adjusted majority count
            adjusted_majority_count = majority_multiplier * majority_count

            # Recompute desired minority_multiplier
            desired_minority_multiplier = adjusted_majority_count / minority_count

            # Enforce minority_multiplier between 1.2 and 3
            minority_multiplier = min(max(desired_minority_multiplier, 1.2), 4)

        print(f"Auto-adjusted multipliers:")
        print(f"majority_multiplier: {majority_multiplier}")
        print(f"minority_multiplier: {minority_multiplier}")
        # **Prepare data for saving:**


    elif args.multiplier_choose == 'from_presetting':
        print("Using preset multipliers without changes.")
        minority_multiplier = args.minority_multiplier
        majority_multiplier = args.majority_multiplier
        adjusted_majority_count = majority_multiplier * majority_count
        adjusted_minority_count = minority_multiplier * minority_count
        print(f"majority_multiplier: {majority_multiplier}")
        print(f"minority_multiplier: {minority_multiplier}")

    else:
        raise ValueError("Invalid option for --multiplier_choose. Choose 'auto' or 'from_presetting'.")

    data = {
        'Class': ['Majority', 'Minority'],
        'Adjusted Multiplier': [majority_multiplier, minority_multiplier],
        'Original Count': [majority_count, minority_count],
        'Adjusted Count': [adjusted_majority_count, adjusted_minority_count]
    }
    df = pd.DataFrame(data)

    # **Save to Excel:**
    df.to_excel(os.path.join(save_path, f'{file_prefix}_multiplier_adjustments.xlsx'), index=False)
    print(f"Adjusted multiplier data has been saved to 'multiplier_adjustments.xlsx'.")

    return majority_multiplier, minority_multiplier


if VAE_augmentation_used:
    majority_multiplier, minority_multiplier = minmaxclass_auto_count(label_counts, args, save_path, file_prefix)


###交叉训练
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(os.path.join(save_path, 'cross_validation_training_log.log')), logging.StreamHandler()])
logger = logging.getLogger()


# 初始化存储这些指标的列表
auc_scores = []
mcc_scores = []
f1_scores = []
recall_scores = []
accuracy_scores = []
precision_scores = []
fpr_list = []
tpr_list = []

precision_list = []
recall_list = []
ap_score_list =[]
fold = 0
# 创建用于存储索引的字典
fold_indices = {}



# 进行交叉验证
for train_index, val_index in skf.split(scFoundation_Embedding_adain, scFoundation_Embedding_info[drug_label_choose]):

    ##在每一个fold的初始重新加载model_pretraining
    model_pretraining = ContinuousResidualVAE(input_dim=input_dim, hidden_dim_layer0=hidden_dim_layer0,
                                              Encoder_layer_dims=Encoder_layer_dims,
                                              Decoder_layer_dims=Decoder_layer_dims,
                                              hidden_dim_layer_out_Z=hidden_dim_layer_out_Z, z_dim=z_dim,
                                              loss_type='MSE',
                                              reduction='mean').to(device)

    model_pretraining.load_state_dict(
        torch.load(os.path.join(path_to_add, model_parameters_file_pretraining), map_location=device))



    ##训练初始冷冻Encoder权重
    freeze_layers(model_pretraining)

    fold_indices[f'Fold {fold + 1}'] = {
        'train_index': train_index,
        'val_index': val_index
    }



    # 分割数据
    X_train, X_val = scFoundation_Embedding_adain[train_index], scFoundation_Embedding_adain[val_index]
    y_train, y_val = scFoundation_Embedding_info[drug_label_choose].iloc[train_index], scFoundation_Embedding_info[drug_label_choose].iloc[val_index]


    if SMOTE_used:
        if VAE_augmentation_used:
            raise ValueError("SMOTE and VAE augmentation should not be used together")
        X_train, y_train = smote_tomek_preprocess(X_train, y_train)
        print(X_train.shape)

    X_train_tensor = torch.Tensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train.values).to(device)



    if VAE_augmentation_used:
        # 统计类别分布
        counter_before_VAE = Counter(y_train)
        print("VAE数据增强前的类别分布：", counter_before_VAE)
        X_augmented, y_augmented = augment_data_with_decoder(model_pretraining, X_train, y_train, minority_multiplier = minority_multiplier, majority_multiplier = majority_multiplier)


        X_train_full = torch.cat([X_train_tensor, X_augmented], dim=0)
        y_train_full = torch.cat([y_train_tensor, y_augmented], dim=0)

        # 统计类别分布
        y_augmented_for_report = y_train_full.cpu().numpy()

        counter_after_VAE = Counter(y_augmented_for_report)

        print("VAE数据增强后的类别分布：", counter_after_VAE)

        # **获取完整的训练标签，包括增强数据的标签**
        # y_train_np = y_train_full.cpu().numpy()

        # print(X_train_full.shape)

        # 将数据转换为 NumPy 数组
        X_train_np = X_train_full.cpu().numpy()
        y_train_np = y_train_full.cpu().numpy()

        # 统计类别分布
        counter_before = Counter(y_train_np)
        print("Tomlink转换前的类别分布：", counter_before)

        # 打印总样本数量
        print("Tomlink转换前的总样本数量：", len(y_train_np))

        # 应用 Tomek Links
        tl = TomekLinks(sampling_strategy='auto')
        X_resampled_np, y_resampled_np = tl.fit_resample(X_train_np, y_train_np)
        # 统计转换后的类别分布
        counter_after = Counter(y_resampled_np)
        print("Tomlink转换后的类别分布：", counter_after)

        # 打印总样本数量
        print("Tomlink转换后的总样本数量：", len(y_resampled_np))

        # 转换回 Tensor
        X_resampled_tensor = torch.from_numpy(X_resampled_np).float().to(device)
        y_resampled_tensor = torch.from_numpy(y_resampled_np).long().to(device)

        train_dataset = TensorDataset(X_resampled_tensor, y_resampled_tensor)
        y_train_np = y_resampled_np

    else:
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # **如果未进行数据增强，使用原始的训练标签**
        y_train_np = y_train_tensor.cpu().numpy()


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)

    ##val
    X_val_tensor = torch.Tensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val.values).to(device)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False)

    # 获取所有类别的标签
    classes = np.unique(y_train_np)

    # 计算类别权重，基于增强后的数据比例
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=classes,
                                         y=y_train_np)

    # 将类别权重转换为Tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)


    ##define model
    model = VAEclassification(model_pretraining, z_dim, class_num).to(device)


    # 训练模型
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=int(num_epochs / 4), eta_min=learning_rate * (1e-2))
    # 损失函数和优化器
    # 假设class_weights已经根据增强后的数据计算得到
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(num_epochs):

        if post_training:
            # 当训练进入最后10轮时解冻层
            if epoch == num_epochs - post_training_epoch_num:
                unfreeze_layers(model_pretraining)

        train_loss = train(model, train_loader, optimizer,criterion)
        val_loss = validate(model, val_loader, criterion)


        message = f" Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"

        logger.info(message)
        print(message)

        ## update learning rate
        scheduler.step()

        # 在最后一个epoch也保存一次
        if epoch + 1 == num_epochs:
            checkpoint_path = os.path.join(save_path, f'checkpoint_fold{fold + 1}_final_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)

    predictions, targets = validate_ACC_final_epoch(model, val_loader)

    # 计算各种指标
    targets = [int(x) for x in targets]
    predictions_prob = [float(x) for x in predictions]
    predictions_label = [1 if x > 0.5 else 0 for x in predictions_prob]

    current_auc = roc_auc_score(targets, predictions_prob)
    current_mcc = matthews_corrcoef(targets, predictions_label)
    current_f1 = f1_score(targets, predictions_label)
    current_recall = recall_score(targets, predictions_label)
    current_accuracy = accuracy_score(targets, predictions_label)
    current_precision = precision_score(targets, predictions_label)

    # 存储每个fold的指标
    auc_scores.append(current_auc)
    mcc_scores.append(current_mcc)
    f1_scores.append(current_f1)
    recall_scores.append(current_recall)
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

    # 为ROC曲线存储FPR和TPR
    fpr, tpr, _ = roc_curve(targets, predictions_prob)
    fpr_list.append(fpr)
    tpr_list.append(tpr)

    # 为PR曲线存储precision和recall
    precision, recall, _ = precision_recall_curve(targets, predictions_prob)
    ap_score = average_precision_score(targets, predictions_prob)
    precision_list.append(precision)
    recall_list.append(recall)
    ap_score_list.append(ap_score)

    fold += 1



# 将 fold_indices 字典保存到 JSON 文件
# 假设 fold_indices 字典中包含 NumPy 数组，我们需要将它们转换为列表
for fold in fold_indices:
    fold_indices[fold]['train_index'] = fold_indices[fold]['train_index'].tolist()  # 转换为列表
    fold_indices[fold]['val_index'] = fold_indices[fold]['val_index'].tolist()  # 转换为列表

with open(os.path.join(save_path,f'{file_prefix}_fold_train_val_indices.json'), 'w') as file:
    json.dump(fold_indices, file, indent=4)  # 使用 indent 参数美化输出格式，使其更易于阅读

# # 读取 JSON 文件中的数据
# with open('fold_indices.json', 'r') as file:
#     loaded_indices = json.load(file)



# Function to extract log data from a logger file
def extract_log_data(file_path):
    with open(file_path, 'r') as file:
        log_data = file.readlines()

    data = []
    pattern = re.compile(r"Fold (\d+), Epoch (\d+)/(\d+), Train Loss: ([\d.]+), Val Loss: ([\d.]+)")

    for line in log_data:
        match = pattern.search(line)
        if match:
            fold = int(match.group(1))
            epoch = int(match.group(2))
            train_loss = float(match.group(4))
            val_loss = float(match.group(5))
            data.append([fold, epoch, train_loss, val_loss])

    return pd.DataFrame(data, columns=['Fold', 'Epoch', 'Train Loss', 'Val Loss'])


# Path to the logger file
file_path = (os.path.join(save_path, 'cross_validation_training_log.log') ) # Replace with the actual path to your logger file

# Extract log data
df = extract_log_data(file_path)

# Remove the first epoch from each fold
df = df[df['Epoch'] > epoch_start_for_loss_plot_only]
print('epoch_start_for_loss_plot_only:', epoch_start_for_loss_plot_only)
# Calculate mean and standard deviation for each epoch across folds
train_loss_mean = df.groupby('Epoch')['Train Loss'].mean()
train_loss_std = df.groupby('Epoch')['Train Loss'].std()
val_loss_mean = df.groupby('Epoch')['Val Loss'].mean()
val_loss_std = df.groupby('Epoch')['Val Loss'].std()

# Plotting the train and validation loss curves
plt.figure(figsize=(12, 6))

# Plot Train Loss
plt.subplot(1, 2, 1)
plt.plot(train_loss_mean.index, train_loss_mean, label='Train Loss')
plt.fill_between(train_loss_mean.index, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Validation Loss
plt.subplot(1, 2, 2)
plt.plot(val_loss_mean.index, val_loss_mean, label='Validation Loss', color='orange')
plt.fill_between(val_loss_mean.index, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2,
                 color='orange')

plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Save the plot as a PDF
plt.savefig(os.path.join(save_path,f"{file_prefix}model_Loss_plot.pdf"),dpi = 1000, format='pdf')


# Plotting the combined train and validation loss curves
plt.figure(figsize=(8, 6))

plt.plot(train_loss_mean.index, train_loss_mean, label='Train Loss')
plt.fill_between(train_loss_mean.index, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.2)
plt.plot(val_loss_mean.index, val_loss_mean, label='Validation Loss', color='orange')
plt.fill_between(val_loss_mean.index, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2, color='orange')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()

# Save the plot as a PDF
plt.savefig(os.path.join(save_path,f"{file_prefix}combined_model_Loss_plot.pdf"),dpi = 1000, format='pdf')



# Set custom colors for plotting (using a clean scientific color palette)
colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c'])  # Blue, Orange, Green, common in SCI-style plots

# Initialize the plot
plt.figure(figsize=(8, 6))

# Plot the ROC curve for each fold
for i, (fpr, tpr, color) in enumerate(zip(fpr_list, tpr_list, colors)):
    plt.plot(fpr, tpr, color=color, lw=1.5, label=f'ROC fold {i+1} (AUC = {auc_scores[i]:.4f})')

# # Compute the mean ROC curve (averaging the folds)

# Plot the diagonal line for random predictions
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)


plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for each fold with Mean ROC', fontsize=14)
plt.legend(loc="lower right", fontsize=10)

# 保存图像到文件
plt.savefig(os.path.join(save_path,f"{file_prefix}fold_10_auc.pdf"),dpi = 1000, format='pdf')



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
for i, (recall, precision, ap_score,color) in enumerate(zip(recall_list, precision_list,ap_score_list ,colors)):
    plt.plot(recall, precision, color=color, lw=1.5,label='Fold {0} PR Curve (AP={1:0.4f})'.format(i+1, ap_score))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Each Fold')
plt.legend(loc="upper right", fontsize=10)
# 保存图像到文件
plt.savefig(os.path.join(save_path,f"{file_prefix}fold_10_PR.pdf"),dpi = 1000, format='pdf')
