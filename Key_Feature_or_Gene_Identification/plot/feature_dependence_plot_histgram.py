# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:47:16 2024

@author: qq102
"""

import configparser
from protloc_mex1.SHAP_plus import FeaturePlot
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import anndata as ad

import argparse

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
parser.add_argument("--open_path_feature_attribution", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data', help="Path to open conference data.")


parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")


# parser.add_argument('--ID', type=str, default='patient_id', help='label of classes.')

parser.add_argument('--seed_set', type=int, default=42, help='Number of random seed.')

parser.add_argument('--gene_choose_name', type=str, default='HSP90AA1', help='label of classes.')
parser.add_argument('--feature_name_to_gene', type=str)


parser.add_argument('--jointplot_kwargs_bins', type=int, default=50, help='Number of bins set.')

parser.add_argument('--depleted_ID_len', type=int, default=0, help='Number of the drop the last non-attributable value column.')

# parser.add_argument(
#     "--shap_calculate_type",
#     type=str,
#     nargs='+',  # Allows multiple values
#     default=["fa34os", "fa34oc"],  # Default includes both values
#     help="Select calculation types; multiple values can be included, e.g., fa34os fa34oc"
# )


parser.add_argument(
    "--shap_indicate_feature",
    type=str,
    nargs='+',  # Allows multiple values
    default=["A1BG", "TP53"]  # Default includes both values
)



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


# 设置PDF字体参数
plt.rcParams['pdf.fonttype'] = 42  # 确保嵌入字体而不是将其转换为路径

print(args.shap_indicate_feature)


##无torch版
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


set_seed(args.seed_set)



# 检查目录是否存在，如果不存在则创建
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)



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
# print('info data shape: ',scFoundation_Embedding_info.shape)






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




# Function to load all CSV and XLSX files in a directory into a dictionary
def load_files_to_dict(directory):
    file_dict = {}
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        file_name, file_extension = os.path.splitext(file)
        if file_extension == '.csv':
            file_dict[file_name] = pd.read_csv(file_path)
        elif file_extension == '.xlsx':
            file_dict[file_name] = pd.read_excel(file_path)


    return file_dict



type_data_all=load_files_to_dict(args.open_path_feature_attribution)



for i in type_data_all.keys():
    type_data_all[i]=type_data_all[i].iloc[:,:len(type_data_all[i].columns)-args.depleted_ID_len]



print(f"绘图数据点数量: {len(data_df)}")





test_plot = FeaturePlot(data_df, type_data_all)


test_plot.feature_hist_plot(args.shap_indicate_feature,
                            args.save_path+'/',
                            file_name='',
                            png_plot=False,jointplot_kwargs={'bins': args.jointplot_kwargs_bins},
                          marg_x_kwargs={'bins': 20,'edgecolor':'black', 'linewidth':1},
                          marg_y_kwargs={'bins': 20,'edgecolor':'black', 'linewidth':1})

