# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:44:41 2024

@author: qq102
"""




import scanpy as sc
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
from protloc_mex1.SHAP_plus import FeaturePlot
from tqdm import tqdm
# from protloc_mex1.IG_calculator import IntegratedGradientsCalculator
import anndata as ad

import seaborn as sns

import argparse
import torch.nn.init as init
from torch.nn.utils import spectral_norm
import json
import numpy as np
import random

from scipy.stats import wilcoxon, mannwhitneyu, trim_mean

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


parser.add_argument("--model_parameters_file", type=str, default='model_parameters.pth', help="File name for storing model parameters.")


parser.add_argument('--label_mapping', type=str, default='{"tos": 0, "toR": 1}',
                    help='JSON string for label mapping e.g. \'{"tos": 0, "toR": 1}\'')

# parser.add_argument('--drug_label_choose', type=str, default='label', help='label of classes.')
parser.add_argument('--ID', type=str, default='patient_id', help='label of classes.')
parser.add_argument('--cell_attribution', type=str, default='label', help='label of classes.')
parser.add_argument('--seed_set', type=int, default=42, help='Number of random seed.')

parser.add_argument('--gene_choose_name', type=str, default='HSP90AA1', help='label of classes.')


parser.add_argument('--feature_name_to_gene', type=str)



# 添加模式参数，类型为字符串，默认值为 'gene_mode'
parser.add_argument(
        '--mode',
        type=str,
        default='gene_mode',
        choices=['gene_mode', 'feature_mode'],
        help='选择运行模式： "gene_mode" 表示基因模式，"feature_mode" 表示特征模式。默认为 "gene_mode"。'
    )

parser.add_argument('--feature_name', type=str,default='scFoundation')

# Parse the arguments
args = parser.parse_args()



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(args.seed_set)





# 检查目录是否存在，如果不存在则创建
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)



print(f"Gene ID: {args.ID}")



# 设置PDF字体参数
plt.rcParams['pdf.fonttype'] = 42  # 确保嵌入字体而不是将其转换为路径
#plt.rcParams['pdf.use14corefonts'] = True  # 使用14种核心字体，避免自定义字体带来的问题





##input_data
filenames = os.listdir(args.open_path)
for filename in filenames:
    # scFoundation_Embedding = np.load(os.path.join(open_path, filename))
    full_path = os.path.join(args.open_path, filename)

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
    elif file_extension == '.h5ad':
        adata = ad.read_h5ad(full_path)
        scFoundation_Embedding = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X  # 确保是NumPy数组
        print("Loaded .h5ad file:", scFoundation_Embedding)
    else:
        print("Unsupported file format")

# 检查数据是否含有NaN
print("DATA is containing NA?: ", np.isnan(scFoundation_Embedding).any())
print('Embedding data shape: ',scFoundation_Embedding.shape)
print('info data shape: ',scFoundation_Embedding_info.shape)


##无标签和ID，可通过model_inference后读取

#
# num_features = scFoundation_Embedding.shape[1]


if args.mode =='gene_mode':

    # 使用绝对路径读入 Excel 文件
    file_path = args.feature_name_to_gene

    data_feature_to_gene = pd.read_excel(file_path)

    # 从第一列生成 feature_names 列表
    feature_names = data_feature_to_gene.iloc[:, 0].tolist()
    # 直接将标签信息添加到 data_df，不设置索引
    data_df = pd.DataFrame(scFoundation_Embedding, columns=feature_names)
elif args.mode =='feature_mode':
    data_df = pd.DataFrame(scFoundation_Embedding)

    # Determine the number of latent dimensions
    latent_dim = data_df.shape[1]

    # Rename the columns to latent_z_num
    data_df.columns = [f'{args.feature_name}_{i}' for i in range(latent_dim)]

# 读取标签信息并合并



# scFoundation_Embedding_info.index = scFoundation_Embedding_info.index.astype(int)
# adata.obs = adata.obs.join(scFoundation_Embedding_info, how='left')

# ids = scFoundation_Embedding_info[args.ID]

# 直接将标签信息添加到 data_df，不设置索引
# data_df = pd.DataFrame(scFoundation_Embedding, columns=feature_names)
data_df[args.ID] = scFoundation_Embedding_info[args.ID].values  # 添加ID列用于合并
data_df['label'] = scFoundation_Embedding_info.set_index(args.ID).loc[data_df[args.ID], args.cell_attribution].values

# 检查标签列是否正确合并
print(data_df['label'])


# 提取唯一标签
unique_labels = data_df['label'].unique()
print(f"唯一的标签值: {unique_labels}")

# 验证标签数量
if len(unique_labels) != 2:
    raise ValueError(f"期望有两个唯一标签，但发现 {len(unique_labels)} 个: {unique_labels}. 请检查数据。")

# 定义组标签
group1_label = unique_labels[0]
group2_label = unique_labels[1]

print(f"Group 1 标签: {group1_label}")
print(f"Group 2 标签: {group2_label}")



# 创建小提琴图




plt.figure(figsize=(8, 6))

sns.violinplot(data=data_df, x='label', y = args.gene_choose_name, inner='point', cut=0, scale='width')
#
# # 图形美化
# plt.xlabel("Drug Sensitivity Label")
# plt.ylabel(f"{args.gene_choose_name} Expression")
# plt.title(f"Distribution of {args.gene_choose_name} Expression by Drug Sensitivity Label")
#
#
#
# # 保存为PNG，DPI设为1000
# output_png_path = f"{args.save_path}/{args.file_prefix}_{args.gene_choose_name}_violin_plot.png"
# plt.savefig(output_png_path, dpi=1000, bbox_inches='tight')
# print(f"PNG文件已保存到: {output_png_path}")
#
# # 保存为PDF，应用字体设置
# output_pdf_path = f"{args.save_path}/{args.file_prefix}_{args.gene_choose_name}_violin_plot.pdf"
# plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight')
# print(f"PDF文件已保存到: {output_pdf_path}")


# 分组
group1 = data_df[data_df['label'] == group1_label][args.gene_choose_name]
group2 = data_df[data_df['label'] == group2_label][args.gene_choose_name]

# 选择检验类型
u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

# 计算截尾均值（Trimmed Mean）
trim_prop = 0.1
trim_mean1 = trim_mean(group1, proportiontocut=trim_prop)
trim_mean2 = trim_mean(group2, proportiontocut=trim_prop)

# 计算平均值（Mean）和中位数（Median）
mean1 = group1.mean()
mean2 = group2.mean()
median1 = group1.median()
median2 = group2.median()

# 计算截尾均值差异
mean_diff = trim_mean1 - trim_mean2

print(f"截尾均值 '{group1_label}': {trim_mean1:.3f}")
print(f"截尾均值 '{group2_label}': {trim_mean2:.3f}")
print(f"平均值 '{group1_label}': {mean1:.3f}")
print(f"平均值 '{group2_label}': {mean2:.3f}")
print(f"中位数 '{group1_label}': {median1:.3f}")
print(f"中位数 '{group2_label}': {median2:.3f}")
print(f"截尾均值差异 (ΔTrimMean): {mean_diff:.3f}")

# 创建一个 DataFrame 用于保存每组的统计信息
group_stats_df = pd.DataFrame({
    'Statistic': ['Trimmed_Mean', 'Mean', 'Median'],
    group1_label: [trim_mean1, mean1, median1],
    group2_label: [trim_mean2, mean2, median2]
})

# 创建一个 DataFrame 用于保存 Mann-Whitney U 检验的统计量、p值和截尾均值差异
test_stats_df = pd.DataFrame({
    'Statistic': ['Mann-Whitney U', 'p-value', 'ΔTrimMean'],
    'Value': [u_stat, p_value, mean_diff]
})

# 将两个 DataFrame 保存到同一个 Excel 文件的不同工作表中
excel_output_path = f"{args.save_path}/{args.file_prefix}_{args.gene_choose_name}_statistics.xlsx"
try:
    with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
        group_stats_df.to_excel(writer, sheet_name='Group_Statistics', index=False)
        test_stats_df.to_excel(writer, sheet_name='Test_Statistics', index=False)
    print(f"统计信息已保存到 Excel 文件: {excel_output_path}")
except Exception as e:
    raise IOError(f"无法保存 Excel 文件: {e}")


# 图形美化
plt.xlabel("Drug Sensitivity Label")
plt.ylabel(f"{args.gene_choose_name} Expression")
plt.title(f"Distribution of {args.gene_choose_name} Expression by Drug Sensitivity Label")

# 添加统计信息到图上
# 确定位置
y_max = data_df[args.gene_choose_name].max()
y_min = data_df[args.gene_choose_name].min()
y, h, col = y_max + (y_max - y_min) * 0.05, (y_max - y_min) * 0.02, 'k'

# 绘制连线
plt.plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.5, c=col)
# 添加p值和均值差异文本
plt.text(0.5, y + h, f"p = {p_value:.3e}\nΔTrimMean = {mean_diff:.3f}", ha='center', va='bottom', color=col)

# 保存图形
output_png_path = f"{args.save_path}/{args.file_prefix}_{args.gene_choose_name}_violin_plot.png"
plt.savefig(output_png_path, dpi=1000, bbox_inches='tight')
print(f"PNG文件已保存到: {output_png_path}")

output_pdf_path = f"{args.save_path}/{args.file_prefix}_{args.gene_choose_name}_violin_plot.pdf"
plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight')
print(f"PDF文件已保存到: {output_pdf_path}")


print('run feature_aggregation_value_conduct_Tree_SHAP.py success, feature importance value and importance rank are deploted in ./output')




